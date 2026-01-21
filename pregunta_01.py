import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Carga de datos
# Los archivos contienen informacion de productos, inventario y transacciones
# -----------------------------------------------------------------------------
print("=" * 70)
print("PARTE 1: RETAIL ANALYTICS")
print("=" * 70)

# Cargamos los tres archivos necesarios
maestro_productos = pd.read_csv("data/maestro_productos.csv")
inventario_diario = pd.read_csv("data/inventario_diario.csv")
transacciones = pd.read_csv("data/transacciones_ventas.csv")

# Convertimos las columnas numericas (pueden venir como texto en algunos CSVs)
columnas_numericas_trans = ['unidades_vendidas', 'precio_unitario_venta', 'costo_unitario', 
                            'precio_lista_original', 'monto_descuento_unitario']
for col in columnas_numericas_trans:
    if col in transacciones.columns:
        transacciones[col] = pd.to_numeric(transacciones[col], errors='coerce')

columnas_numericas_inv = ['cantidad_stock', 'valor_inventario_costo']
for col in columnas_numericas_inv:
    if col in inventario_diario.columns:
        inventario_diario[col] = pd.to_numeric(inventario_diario[col], errors='coerce')

# Eliminamos filas con valores nulos en columnas criticas
transacciones = transacciones.dropna(subset=['unidades_vendidas', 'precio_unitario_venta', 'costo_unitario'])
inventario_diario = inventario_diario.dropna(subset=['valor_inventario_costo'])

print("\n[DATOS] Productos cargados:")
print(f"   - Total de productos unicos: {maestro_productos['product_id'].nunique()}")
print(f"   - Categorias disponibles: {maestro_productos['categoria'].unique().tolist()}")

# -----------------------------------------------------------------------------
# PREGUNTA 1. Cálculo de KPIs.
# -----------------------------------------------------------------------------
# PREGUNTA 1.1: Cero Censurado (Out-Of-Stock) y Sustituciones de SKUs
# Analisis de como estos fenomenos afectan la integridad de los datos historicos
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PREGUNTA 1.1: CERO CENSURADO (OUT-OF-STOCK) Y SUSTITUCIONES DE SKUs")
print("=" * 70)

print("""
[ANALISIS] IMPACTO EN LA INTEGRIDAD DE LOS ANALISIS HISTORICOS
----------------------------------------------------------------------

Los fenomenos de Cero Censurado (Out-Of-Stock) y Sustituciones de SKUs
representan serios desafios para la calidad de los analisis historicos
en retail. A continuacion se explica cada uno:

1. CERO CENSURADO (OUT-OF-STOCK / QUIEBRE DE STOCK)
   -----------------------------------------------
   
   DEFINICION: Ocurre cuando un producto no esta disponible para la venta
   debido a falta de inventario. Durante este periodo, las ventas registradas
   son CERO, pero esto NO refleja la demanda real del producto.
   
   IMPACTO EN LOS ANALISIS:
   
   * SUBESTIMACION DE DEMANDA: Los modelos de pronostico aprenden de datos
     historicos. Si hay periodos con ventas = 0 por quiebre, el modelo
     subestimara la demanda real del producto.
   
   * SESGO EN METRICAS DE ROTACION: El GMROI y otras metricas de eficiencia
     se calculan con ventas historicas. Los quiebres distorsionan estas
     metricas haciendolas parecer peores de lo que realmente son.
   
   * PERDIDA DE VENTAS NO CONTABILIZADA: La venta perdida durante el quiebre
     no se registra en ningun lugar, generando una "demanda invisible".
   
   * ESTACIONALIDAD DISTORSIONADA: Si un quiebre coincide con un pico de
     demanda (ej: Navidad), los patrones estacionales quedan mal calibrados.

2. SUSTITUCIONES DE SKUs
   ----------------------
   
   DEFINICION: Cuando un cliente no encuentra el producto deseado, puede
   comprar un sustituto similar. Esto infla las ventas del sustituto y
   oculta la demanda real de ambos productos.
   
   IMPACTO EN LOS ANALISIS:
   
   * INFLACION ARTIFICIAL: El producto sustituto muestra ventas mas altas
     de lo que tendria en condiciones normales, llevando a sobrestock futuro.
   
   * CORRELACION ESPURIA: Se crean correlaciones falsas entre productos que
     en realidad son sustitutos, no complementarios.
   
   * CANIBALIZACION OCULTA: Es dificil distinguir si las ventas altas de un
     producto son por demanda genuina o por sustitucion forzada.
   
   * ERRORES EN PREDICCION: Los modelos ML pueden aprender patrones incorrectos
     basados en sustituciones temporales, no en preferencias reales.

----------------------------------------------------------------------
MEDIDAS DE INGENIERIA DE DATOS RECOMENDADAS
----------------------------------------------------------------------

1. DETECCION Y MARCADO DE QUIEBRES DE STOCK
   
   * Crear una variable indicadora 'flag_oos' (Out-Of-Stock) que identifique
     dias donde el inventario fue cero o cercano a cero.
   * Unir transacciones con inventario diario para detectar periodos sin stock.
   * Marcar estos registros para excluirlos o tratarlos especialmente en
     modelos de pronostico.

2. ESTIMACION DE DEMANDA CENSURADA
   
   * Usar modelos de demanda censurada (Tobit) que ajustan por la truncacion.
   * Implementar tecnicas de imputacion basadas en periodos similares con stock.
   * Calcular "demanda no satisfecha" usando tasas de conversion historicas.

3. DETECCION DE SUSTITUCIONES
   
   * Analizar picos inusuales en productos cuando otros relacionados estan OOS.
   * Crear matrices de sustitucion basadas en elasticidad cruzada de precios.
   * Implementar reglas de negocio para identificar pares de productos
     sustitutos (misma categoria, precio similar, marca similar).

4. LIMPIEZA Y AJUSTE DE DATOS
   
   * Separar periodos "limpios" de periodos "contaminados" para entrenar modelos.
   * Crear features adicionales: 'dias_desde_ultimo_quiebre', 'productos_oos_categoria'.
   * Implementar pipelines de calidad de datos que reporten anomalias.

5. MONITOREO Y ALERTAS
   
   * Dashboards de salud de inventario que alerten sobre quiebres frecuentes.
   * KPIs de "tasa de disponibilidad" por categoria y producto.
   * Reportes de productos con patron sospechoso de sustitucion.
""")

# Analisis practico: Detectamos posibles quiebres de stock en los datos
print("[INFO] Deteccion de posibles quiebres de stock en los datos:")
print("-" * 70)

# Buscamos productos con inventario = 0
productos_sin_stock = inventario_diario[inventario_diario['cantidad_stock'] == 0]
if len(productos_sin_stock) > 0:
    productos_afectados = productos_sin_stock['product_id'].nunique()
    dias_afectados = productos_sin_stock['fecha'].nunique() if 'fecha' in productos_sin_stock.columns else "N/A"
    print(f"   - Registros con stock = 0 detectados: {len(productos_sin_stock):,}")
    print(f"   - Productos unicos afectados: {productos_afectados:,}")
    print(f"   - Dias con al menos un quiebre: {dias_afectados}")
else:
    print("   - No se detectaron registros con stock = 0 en los datos.")

# -----------------------------------------------------------------------------
# PREGUNTA 1.2: Calculo de GMROI por Categoria
# GMROI = Margen Bruto / Inventario Promedio a Costo
# Esta metrica nos indica cuantos pesos de margen genera cada peso invertido
# en inventario
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PREGUNTA 1.2: GMROI (Gross Margin Return on Investment)")
print("=" * 70)

# Calculamos las metricas de ventas necesarias
# Ingreso total = unidades vendidas x precio de venta
transacciones['ingreso_total'] = (
    transacciones['unidades_vendidas'] * transacciones['precio_unitario_venta']
)

# Costo total = unidades vendidas x costo unitario
transacciones['costo_total'] = (
    transacciones['unidades_vendidas'] * transacciones['costo_unitario']
)

# Margen bruto = Ingreso - Costo
transacciones['margen_bruto'] = (
    transacciones['ingreso_total'] - transacciones['costo_total']
)

# Agrupamos las ventas por categoria para obtener totales
ventas_por_categoria = transacciones.groupby('categoria').agg({
    'ingreso_total': 'sum',
    'costo_total': 'sum',
    'margen_bruto': 'sum'
}).reset_index()

# Calculamos el inventario promedio a costo por categoria
# Esto representa la inversion promedio en stock
inventario_promedio = inventario_diario.groupby('categoria').agg({
    'valor_inventario_costo': 'mean'
}).reset_index()
inventario_promedio.columns = ['categoria', 'inventario_promedio_costo']

# Combinamos los datos para calcular el GMROI
analisis_gmroi = pd.merge(
    ventas_por_categoria, 
    inventario_promedio, 
    on='categoria'
)

# GMROI = Margen Bruto Total / Inventario Promedio a Costo
analisis_gmroi['gmroi'] = (
    analisis_gmroi['margen_bruto'] / analisis_gmroi['inventario_promedio_costo']
)

# Calculamos tambien el porcentaje de margen bruto para el analisis
analisis_gmroi['margen_bruto_pct'] = (
    analisis_gmroi['margen_bruto'] / analisis_gmroi['ingreso_total'] * 100
)

print("\n[RESULTADOS] GMROI por Categoria:")
print("-" * 70)
gmroi_display = analisis_gmroi[['categoria', 'margen_bruto', 'inventario_promedio_costo', 'gmroi', 'margen_bruto_pct']].copy()
gmroi_display.columns = ['Categoria', 'Margen Bruto ($)', 'Inv. Promedio ($)', 'GMROI', 'Margen (%)']
gmroi_display = gmroi_display.sort_values('GMROI', ascending=False)
print(gmroi_display.to_string(index=False))

# -----------------------------------------------------------------------------
# Respuesta a la pregunta: Por que un producto con margen porcentual bajo
# puede tener un GMROI superior a uno de margen alto?
# -----------------------------------------------------------------------------
print("\n" + "-" * 70)
print("[ANALISIS] Por que un producto con margen bajo puede tener GMROI alto?")
print("-" * 70)
print("""
La respuesta se encuentra en la rotacion del inventario. El GMROI combina dos
factores fundamentales:

   GMROI = Margen Bruto % x Rotacion de Inventario

Un producto con margen porcentual bajo puede generar mas retorno si:

   1. ALTA ROTACION: Si el producto se vende rapidamente, el inventario
      promedio es bajo, lo que aumenta el denominador del GMROI.
      
   2. MENOR CAPITAL INMOVILIZADO: Al venderse mas rapido, se libera capital
      para reinvertir, generando mas ciclos de ganancia en el mismo periodo.
      
   3. EFECTO VOLUMEN: Un margen pequeno multiplicado muchas veces supera
      a un margen alto que se concreta pocas veces.

Ejemplo practico:
   - Producto A: Margen 50%, rota 2 veces/ano -> GMROI = 1.0
   - Producto B: Margen 10%, rota 15 veces/ano -> GMROI = 1.5

Conclusion: La velocidad a la que convertimos inventario en ventas es tan
importante como el margen que obtenemos en cada venta.
""")

# -----------------------------------------------------------------------------
# PREGUNTA 1.3: Calculo del Porcentaje de Markdown
# Markdown = Descuento aplicado sobre el precio original
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PREGUNTA 1.3: PORCENTAJE DE MARKDOWN POR CATEGORIA")
print("=" * 70)

# Calculamos el markdown (descuento) como porcentaje del precio original
transacciones['markdown_pct'] = (
    transacciones['monto_descuento_unitario'] / 
    transacciones['precio_lista_original'] * 100
)

# Ponderamos el markdown por las unidades vendidas para obtener un promedio
# mas representativo (productos que se venden mas tienen mas peso)
transacciones['markdown_ponderado'] = (
    transacciones['markdown_pct'] * transacciones['unidades_vendidas']
)

markdown_por_categoria = transacciones.groupby('categoria').agg({
    'markdown_ponderado': 'sum',
    'unidades_vendidas': 'sum'
}).reset_index()

markdown_por_categoria['markdown_promedio_pct'] = (
    markdown_por_categoria['markdown_ponderado'] / 
    markdown_por_categoria['unidades_vendidas']
)

print("\n[RESULTADOS] Porcentaje de Markdown por Categoria:")
print("-" * 70)
markdown_display = markdown_por_categoria[['categoria', 'markdown_promedio_pct']].copy()
markdown_display.columns = ['Categoria', 'Markdown Promedio (%)']
markdown_display = markdown_display.sort_values('Markdown Promedio (%)', ascending=False)
print(markdown_display.to_string(index=False))

# -----------------------------------------------------------------------------
# Combinamos GMROI con Markdown para analisis integral
# -----------------------------------------------------------------------------
analisis_completo = pd.merge(
    analisis_gmroi[['categoria', 'gmroi', 'margen_bruto_pct']], 
    markdown_por_categoria[['categoria', 'markdown_promedio_pct']], 
    on='categoria'
)

# Identificamos categorias con GMROI alto pero Markdown > 20%
print("\n" + "-" * 70)
print("[ALERTA] ANALISIS: Categorias con GMROI alto pero Markdown > 20%")
print("-" * 70)

categorias_riesgo = analisis_completo[
    (analisis_completo['gmroi'] > analisis_completo['gmroi'].median()) & 
    (analisis_completo['markdown_promedio_pct'] > 20)
]

if len(categorias_riesgo) > 0:
    print("\nCategorias identificadas:")
    for _, row in categorias_riesgo.iterrows():
        print(f"   - {row['categoria']}: GMROI={row['gmroi']:.2f}, Markdown={row['markdown_promedio_pct']:.1f}%")
else:
    print("\n   No se encontraron categorias que cumplan ambos criterios.")
    print("   Mostrando categorias con mayor markdown para analisis:")
    top_markdown = analisis_completo.nlargest(3, 'markdown_promedio_pct')
    for _, row in top_markdown.iterrows():
        print(f"   - {row['categoria']}: GMROI={row['gmroi']:.2f}, Markdown={row['markdown_promedio_pct']:.1f}%")

print("\n[ANALISIS] PROBLEMAS QUE PODRIA ESTAR OCULTANDO UN GMROI ALTO CON MARKDOWN > 20%:")
print("-" * 70)
print("""
Cuando observamos un GMROI aparentemente saludable junto con un markdown
excesivo (mayor al 20%), debemos investigar posibles problemas subyacentes:

   1. SOBRESTOCK Y COMPRAS EXCESIVAS
      Los altos descuentos podrian indicar que se compro mas mercaderia de la
      necesaria y ahora se esta liquidando para liberar capital. El GMROI puede
      verse bien temporalmente por la alta rotacion forzada.

   2. PROBLEMAS DE PRONOSTICO DE DEMANDA
      Un markdown elevado sugiere que los precios originales no reflejan
      correctamente la disposicion a pagar del cliente, lo que indica fallas
      en el pricing inicial o errores en la prediccion de demanda.

   3. COMPETENCIA AGRESIVA EN PRECIOS
      El mercado podria estar forzando reducciones de precio para mantener
      competitividad. Aunque las ventas se mantienen, los margenes reales
      estan erosionandose.

   4. DETERIORO O OBSOLESCENCIA
      Productos cercanos a vencimiento o que se estan volviendo obsoletos
      requieren descuentos agresivos para venderse, ocultando problemas
      de gestion de inventario.

   5. INFLACION DE GMROI TEMPORAL
      Un markdown alto puede inflar artificialmente el GMROI a corto plazo
      (mas ventas = menos inventario promedio), pero esto no es sostenible
      y erosiona la rentabilidad real del negocio.

RECOMENDACION: Analizar la tendencia historica del markdown y correlacionarla
con los niveles de inventario inicial de cada temporada para identificar
si existe un patron de sobrecompra sistematica.
""")

# -----------------------------------------------------------------------------
# Tabla resumen final
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("[RESUMEN] EJECUTIVO - ANALISIS RETAIL")
print("=" * 70)
resumen = analisis_completo.copy()
resumen.columns = ['Categoria', 'GMROI', 'Margen Bruto (%)', 'Markdown (%)']
resumen = resumen.sort_values('GMROI', ascending=False)
print(resumen.round(2).to_string(index=False))
print("\n" + "=" * 70)
