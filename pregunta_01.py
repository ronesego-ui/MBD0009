# -*- coding: utf-8 -*-

# Instalar librerias pendientes
#pip install rapidfuzz

#librerias
import pandas as pd
import numpy as np
import re
import unicodedata
from rapidfuzz import process, fuzz

#Cargar archivos
#inv = pd.read_csv('C:/Users/felip/Desktop/Magíster/8. Marketing y Analítica del Retail/Tarea 1/inventario_diario.csv')
#prod = pd.read_csv('C:/Users/felip/Desktop/Magíster/8. Marketing y Analítica del Retail/Tarea 1/maestro_productos.csv')
#trx = pd.read_csv('C:/Users/felip/Desktop/Magíster/8. Marketing y Analítica del Retail/Tarea 1/transacciones_ventas.csv')

inv = pd.read_csv("data/inventario_diario.csv")
prod = pd.read_csv("data/maestro_productos.csv")
trx = pd.read_csv("data/transacciones_ventas.csv")


#Funciones de limpieza, normalización y relleno de datos
#_________________________________________________________________________

##################### Limpieza de columna fecha ###########################
def limpiar_fecha(df, col_fecha):
    df = df.copy()

    # 1. Eliminar letras y caracteres raros
    df[col_fecha] = (
        df[col_fecha]
        .astype(str)
        .str.replace(r"[a-zA-Z]", "", regex=True)
        .str.replace(r"[^0-9\-\/]", "", regex=True)
        .replace("", pd.NA)
    )

    # Intento 1: dd-mm-aaaa
    fecha_1 = pd.to_datetime(
        df[col_fecha],
        errors="coerce",
        dayfirst=True
    )

    # Intento 2: aaaa-mm-dd
    fecha_2 = pd.to_datetime(
        df[col_fecha],
        errors="coerce",
        yearfirst=True
    )

    # Combinar intentos
    fechas = fecha_1.combine_first(fecha_2)

    # Casos donde solo viene el año (yyyy)
    solo_anio = df[col_fecha].astype(str).str.fullmatch(r"\d{4}")
    fechas.loc[solo_anio] = pd.to_datetime(
        df.loc[solo_anio, col_fecha] + "-01-01",
        errors="coerce"
    )

    # 3. Forzar todos los años a 2025
    fechas.loc[fechas.notna()] = fechas.loc[fechas.notna()].apply(
        lambda x: x.replace(year=2025)
    )

    # 4. Rellenar fechas vacías con la inmediatamente anterior
    fechas = fechas.ffill()

    # Formato final
    df[col_fecha] = fechas.dt.strftime("%d-%m-%Y")

    return df


##################### Limpieza de columna categoria #########################

#------- Normalización -------#

def normalizar_categoria(df, col_categoria):
    df = df.copy()

    def _normalizar(x):
        if pd.isna(x):
            return x

        x = str(x)

        # Corregir encoding roto (MÃ³ → ó)
        try:
            x = x.encode("latin1").decode("utf-8")
        except Exception:
            pass

        # Minúsculas
        x = x.lower()

        # Quitar tildes
        x = unicodedata.normalize("NFKD", x)
        x = "".join(c for c in x if not unicodedata.combining(c))

        # Eliminar caracteres no alfabéticos
        x = re.sub(r"[^a-z\s]", "", x)

        # Normalizar espacios
        x = re.sub(r"\s+", " ", x).strip()

        return x

    df[col_categoria] = df[col_categoria].apply(_normalizar)

    return df

#------- Diccionario palabras correctas -------#

categorias_reales = [
    "despensa",
    "hogar",
    "lenceria",
    "pequenos electrodomesticos",
    "carnes",
    "cuidado del bebe",
    "viaje",
    "bebidas",
    "ropa nino",
    "higiene personal",
    "salud",
    "relojeria",
    "electronica",
    "panaderia",
    "automotriz",
    "informatica",
    "ropa mujer",
    "jardin",
    "pescados",
    "mascotas",
    "camping",
    "papeleria",
    "belleza",
    "limpieza",
    "bisuteria",
    "lacteos",
    "congelados",
    "television y audio",
    "muebles",
    "iluminacion",
    "juguetes",
    "ropa hombre",
    "alimentos",
    "farmacia",
    "decoracion",
    "frutas y verduras",
    "calzado",
    "ferreteria",
    "deportes",
    "accesorios moviles",
    "moda"
]

# ----- Match por similitud ----- #

def match_categoria_con_reales(
    df,
    col_categoria,
    categorias_reales,
    threshold=80
):
    """
    Asigna cada categoría a la mejor coincidencia dentro de categorias_reales
    usando fuzzy matching.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada
    col_categoria : str
        Nombre de la columna de categorías
    categorias_reales : list
        Lista de categorías válidas (normalizadas)
    threshold : int
        Umbral mínimo de similitud (0–100)

    Retorna
    -------
    pd.DataFrame
        DataFrame con la categoría corregida
    """

    df = df.copy()

    def _match(x):
        if pd.isna(x):
            return x

        match = process.extractOne(
            x,
            categorias_reales,
            scorer=fuzz.token_sort_ratio
        )

        if match and match[1] >= threshold:
            return match[0]

        return x  # si no alcanza el umbral, no se reemplaza

    df[col_categoria] = df[col_categoria].apply(_match)

    return df




################ Limpieza de columna precios y cantidad ####################

def limpiar_col_num(df, col):
    """
    Elimina símbolos $ y -, limpia separadores y convierte la columna a numérica.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada
    col : str
        Nombre de la columna a limpiar

    Retorna
    -------
    pd.DataFrame
        DataFrame con la columna convertida a numérica
    """

    df = df.copy()

    def _limpiar(x):
        if pd.isna(x):
            return x

        x = str(x)

        # Eliminar símbolos $ y -
        x = re.sub(r"[\$-]", "", x)

        # Eliminar espacios
        x = x.strip()

        return x

    df[col] = df[col].apply(_limpiar)

    # Convertir a numérico (valores no convertibles → NaN)
    df[col] = pd.to_numeric(df[col], errors="coerce")

    return df



################ Completar tablas con categorías faltantes ####################

import pandas as pd

def construir_prod_completo(
    prod,
    inv,
    trx,
    col_id="product_id",
    col_categoria="categoria",
    col_costo="costo_unitario",
    col_precio_prod="precio_lista",
    col_precio_trx="precio_lista_original"
):
    """
    Construye una tabla prod completa usando prod, inv y trx
    siguiendo reglas de prioridad y sin sobrescribir valores existentes.
    """

    # -------------------------------------------------
    # 1. Copia inicial de prod
    prod_nuevo = prod.copy()

    # -------------------------------------------------
    # 2. Incorporar product_id faltantes desde inv
    inv_ids = inv[col_id].dropna().unique()
    prod_ids = prod_nuevo[col_id].dropna().unique()

    nuevos_inv = inv.loc[
        inv[col_id].isin(set(inv_ids) - set(prod_ids)),
        [col_id, col_categoria]
    ].drop_duplicates(subset=[col_id])

    prod_nuevo = pd.concat([prod_nuevo, nuevos_inv], ignore_index=True)

    # -------------------------------------------------
    # 3. Incorporar product_id faltantes desde trx
    prod_ids = prod_nuevo[col_id].dropna().unique()

    nuevos_trx = trx.loc[
        trx[col_id].isin(set(trx[col_id].dropna()) - set(prod_ids)),
        [col_id, col_categoria]
    ].drop_duplicates(subset=[col_id])

    prod_nuevo = pd.concat([prod_nuevo, nuevos_trx], ignore_index=True)

    # -------------------------------------------------
    # 4. Completar categoría (prioridad: inv → trx)
    mapa_cat_inv = (
        inv[[col_id, col_categoria]]
        .dropna()
        .drop_duplicates(subset=[col_id])
        .set_index(col_id)[col_categoria]
        .to_dict()
    )

    mapa_cat_trx = (
        trx[[col_id, col_categoria]]
        .dropna()
        .drop_duplicates(subset=[col_id])
        .set_index(col_id)[col_categoria]
        .to_dict()
    )

    mask_cat = prod_nuevo[col_categoria].isna() & prod_nuevo[col_id].notna()

    prod_nuevo.loc[mask_cat, col_categoria] = (
        prod_nuevo.loc[mask_cat, col_id]
        .map(mapa_cat_inv)
        .fillna(prod_nuevo.loc[mask_cat, col_id].map(mapa_cat_trx))
    )

    # -------------------------------------------------
    # 5. Completar costo_unitario desde trx
    mapa_costo = (
        trx[[col_id, col_costo]]
        .dropna()
        .drop_duplicates(subset=[col_id])
        .set_index(col_id)[col_costo]
        .to_dict()
    )

    mask_costo = prod_nuevo[col_costo].isna() & prod_nuevo[col_id].notna()

    prod_nuevo.loc[mask_costo, col_costo] = (
        prod_nuevo.loc[mask_costo, col_id].map(mapa_costo)
    )

    # -------------------------------------------------
    # 6. Completar precio_lista desde trx
    mapa_precio = (
        trx[[col_id, col_precio_trx]]
        .dropna()
        .drop_duplicates(subset=[col_id])
        .set_index(col_id)[col_precio_trx]
        .to_dict()
    )

    mask_precio = prod_nuevo[col_precio_prod].isna() & prod_nuevo[col_id].notna()

    prod_nuevo.loc[mask_precio, col_precio_prod] = (
        prod_nuevo.loc[mask_precio, col_id].map(mapa_precio)
    )

    return prod_nuevo




############# Completar tabla transacciones #################

def completar_trx_desde_prod(
    trx,
    prod,
    col_product_id="product_id"
):
    """
    Completa campos faltantes en trx usando la tabla maestra prod
    y luego imputa unidades_vendidas por mediana por categoria.
    """

    trx = trx.copy()
    prod = prod.copy()

    # ---------------------------------
    # Diccionarios desde prod
    # ---------------------------------
    mapa_categoria = (
        prod[[col_product_id, "categoria"]]
        .dropna(subset=[col_product_id, "categoria"])
        .drop_duplicates(col_product_id)
        .set_index(col_product_id)["categoria"]
        .to_dict()
    )

    mapa_precio_lista = (
        prod[[col_product_id, "precio_lista"]]
        .dropna(subset=[col_product_id, "precio_lista"])
        .drop_duplicates(col_product_id)
        .set_index(col_product_id)["precio_lista"]
        .to_dict()
    )

    mapa_costo = (
        prod[[col_product_id, "costo_unitario"]]
        .dropna(subset=[col_product_id, "costo_unitario"])
        .drop_duplicates(col_product_id)
        .set_index(col_product_id)["costo_unitario"]
        .to_dict()
    )

    # ---------------------------------
    # 1. Categoria
    # ---------------------------------
    mask = trx["categoria"].isna() & trx[col_product_id].notna()
    trx.loc[mask, "categoria"] = trx.loc[mask, col_product_id].map(mapa_categoria)

    # ---------------------------------
    # 2. Precio lista original
    # ---------------------------------
    mask = trx["precio_lista_original"].isna() & trx[col_product_id].notna()
    trx.loc[mask, "precio_lista_original"] = (
        trx.loc[mask, col_product_id].map(mapa_precio_lista)
    )

    # ---------------------------------
    # 3. Costo unitario
    # ---------------------------------
    mask = trx["costo_unitario"].isna() & trx[col_product_id].notna()
    trx.loc[mask, "costo_unitario"] = (
        trx.loc[mask, col_product_id].map(mapa_costo)
    )

    # ---------------------------------
    # 4. Precio unitario venta
    # precio_lista_original - monto_descuento_unitario
    # ---------------------------------
    mask = (
        trx["precio_unitario_venta"].isna() &
        trx["precio_lista_original"].notna() &
        trx["monto_descuento_unitario"].notna()
    )

    trx.loc[mask, "precio_unitario_venta"] = (
        trx.loc[mask, "precio_lista_original"]
        - trx.loc[mask, "monto_descuento_unitario"]
    )

    # ---------------------------------
    # 5. Monto descuento unitario
    # precio_lista_original - precio_unitario_venta
    # ---------------------------------
    mask = (
        trx["monto_descuento_unitario"].isna() &
        trx["precio_lista_original"].notna() &
        trx["precio_unitario_venta"].notna()
    )

    trx.loc[mask, "monto_descuento_unitario"] = (
        trx.loc[mask, "precio_lista_original"]
        - trx.loc[mask, "precio_unitario_venta"]
    )

    # =========================================================
    # 6. IMPUTACIÓN: unidades_vendidas por mediana de categoria
    # =========================================================

    # Mediana por categoria (solo valores válidos)
    mediana_por_categoria = (
        trx.groupby("categoria")["unidades_vendidas"]
        .median()
        .round()
        .astype("Int64")
    )

    mask = trx["unidades_vendidas"].isna() & trx["categoria"].notna()

    trx.loc[mask, "unidades_vendidas"] = (
        trx.loc[mask, "categoria"]
        .map(mediana_por_categoria)
    )

    return trx


############# Completar tabla inventario #################

def completar_inv_desde_prod(
    inv,
    prod,
    col_product_id="product_id"
):
    """
    Completa la categoría en inv desde prod usando product_id,
    imputa cantidad_stock por mediana por categoria,
    e imputa valor_inventario_costo cuando esté vacío.
    """

    inv = inv.copy()
    prod = prod.copy()

    # ---------------------------------
    # 1. Diccionario product_id → categoria
    # ---------------------------------
    mapa_categoria = (
        prod[[col_product_id, "categoria"]]
        .dropna(subset=[col_product_id, "categoria"])
        .drop_duplicates(col_product_id)
        .set_index(col_product_id)["categoria"]
        .to_dict()
    )

    # ---------------------------------
    # 2. Completar categoria en inv
    # ---------------------------------
    mask = inv["categoria"].isna() & inv[col_product_id].notna()

    inv.loc[mask, "categoria"] = (
        inv.loc[mask, col_product_id].map(mapa_categoria)
    )

    # ---------------------------------
    # 3. Imputar cantidad_stock por mediana de categoria
    # ---------------------------------
    mediana_stock = (
        inv.groupby("categoria")["cantidad_stock"]
        .median()
        .round()
        .astype("Int64")
    )

    mask = inv["cantidad_stock"].isna() & inv["categoria"].notna()

    inv.loc[mask, "cantidad_stock"] = (
        inv.loc[mask, "categoria"].map(mediana_stock)
    )

    # ---------------------------------
    # 4. Imputar valor_inventario_costo
    #    = cantidad_stock * costo_unitario (desde prod)
    # ---------------------------------
    mapa_costo = (
        prod[[col_product_id, "costo_unitario"]]
        .dropna(subset=[col_product_id, "costo_unitario"])
        .drop_duplicates(col_product_id)
        .set_index(col_product_id)["costo_unitario"]
        .to_dict()
    )

    mask = (
        inv["valor_inventario_costo"].isna()
        & inv["cantidad_stock"].notna()
        & inv[col_product_id].notna()
    )

    inv.loc[mask, "valor_inventario_costo"] = (
        inv.loc[mask, "cantidad_stock"]
        * inv.loc[mask, col_product_id].map(mapa_costo)
    )

    return inv
#_________________________________________________________________________

##### Corregir y normalizar tablas #####

# normalizar tabla inv
inv = limpiar_fecha(inv, "fecha")
inv = normalizar_categoria(inv, "categoria")
inv = match_categoria_con_reales(inv, "categoria", categorias_reales, threshold=40)
inv = limpiar_col_num(inv, "cantidad_stock")
inv = limpiar_col_num(inv, "valor_inventario_costo")

# normalizar tabla trx
trx = limpiar_fecha(trx, "fecha")
trx = normalizar_categoria(trx, "categoria")
trx = match_categoria_con_reales(trx, "categoria", categorias_reales, threshold=40)
trx = limpiar_col_num(trx, "unidades_vendidas")
trx = limpiar_col_num(trx, "precio_unitario_venta")
trx = limpiar_col_num(trx, "precio_lista_original")
trx = limpiar_col_num(trx, "monto_descuento_unitario")
trx = limpiar_col_num(trx, "costo_unitario")

#normalizar tabla prod
prod = normalizar_categoria(prod, "categoria")
prod = match_categoria_con_reales(prod, "categoria", categorias_reales, threshold=40)
prod = limpiar_col_num(prod, "costo_unitario")
prod = limpiar_col_num(prod, "precio_lista")

#_________________________________________________________________________

##### Imputar valores a tablas para cálculos #####

# Completar tabla maestra de productos
prod = construir_prod_completo(prod, inv, trx)

# Completar tabla de transacciones
trx = completar_trx_desde_prod(trx, prod)

#Completar tabla de inventario
inv = completar_inv_desde_prod(inv, prod)

#_________________________________________________________________________

##### Cálculo de GMROI por categoría #####

# =========================
# 1. Preparar TRX
# =========================

trx_gmroi = trx.copy()

# Eliminar filas con información incompleta para el cálculo
trx_gmroi = trx_gmroi.dropna(
    subset=[
        "categoria",
        "unidades_vendidas",
        "precio_unitario_venta",
        "costo_unitario"
    ]
)

# Ventas
trx_gmroi["ventas"] = (
    trx_gmroi["unidades_vendidas"] * trx_gmroi["precio_unitario_venta"]
)

# Costo de ventas
trx_gmroi["costo_ventas"] = (
    trx_gmroi["unidades_vendidas"] * trx_gmroi["costo_unitario"]
)

# Margen bruto
trx_gmroi["margen_bruto"] = (
    trx_gmroi["ventas"] - trx_gmroi["costo_ventas"]
)

# Margen bruto por categoría
margen_por_categoria = (
    trx_gmroi
    .groupby("categoria", as_index=False)["margen_bruto"]
    .sum()
)

# =========================
# 2. Preparar INV
# =========================

inv_gmroi = inv.copy()

# Eliminar filas sin categoría o sin valor de inventario
inv_gmroi = inv_gmroi.dropna(
    subset=[
        "categoria",
        "valor_inventario_costo"
    ]
)

# Inventario promedio a costo por categoría
inventario_promedio = (
    inv_gmroi
    .groupby("categoria", as_index=False)["valor_inventario_costo"]
    .mean()
    .rename(columns={
        "valor_inventario_costo": "inventario_promedio_costo"
    })
)

# =========================
# 3. Calcular GMROI
# =========================

gmroi_categoria = (
    margen_por_categoria
    .merge(inventario_promedio, on="categoria", how="inner")
)

gmroi_categoria["gmroi"] = (
    gmroi_categoria["margen_bruto"]
    / gmroi_categoria["inventario_promedio_costo"]
)

# Limpiar infinitos (por inventario = 0)
gmroi_categoria = gmroi_categoria.replace(
    [np.inf, -np.inf],
    np.nan
)

print(gmroi_categoria)

#_________________________________________________________________________


##### Cálculo de MARKDOWN por categoría #####

trx_md = trx.copy()

# --- Filtrar filas válidas ---
trx_md = trx_md.dropna(
    subset=[
        "categoria",
        "unidades_vendidas",
        "precio_unitario_venta",
        "monto_descuento_unitario"
    ]
)

# Eliminar categorías vacías o solo espacios
trx_md = trx_md[trx_md["categoria"].str.strip() != ""]

# --- Calcular descuentos ---
trx_md["descuentos"] = (
    trx_md["unidades_vendidas"] * trx_md["monto_descuento_unitario"]
)

# --- Calcular ventas brutas (dinero real) ---
trx_md["ventas"] = (
    trx_md["unidades_vendidas"] * trx_md["precio_unitario_venta"]
)

# --- Eliminar filas con ventas <= 0 ---
trx_md = trx_md[trx_md["ventas"] > 0]

# --- Agrupar por categoría ---
markdown_categoria = (
    trx_md
    .groupby("categoria", as_index=False)
    .agg(
        descuentos_totales=("descuentos", "sum"),
        ventas_brutas=("ventas", "sum")
    )
)

# --- Proteger cálculo final ---
markdown_categoria = markdown_categoria[
    markdown_categoria["ventas_brutas"] > 0
]

# --- Calcular Markdown ---
markdown_categoria["markdown"] = (
    markdown_categoria["descuentos_totales"]
    / markdown_categoria["ventas_brutas"]
)

# --- Markdown en porcentaje  ---
markdown_categoria["markdown_pct"] = (
    markdown_categoria["markdown"] * 100
)

print(markdown_categoria)
