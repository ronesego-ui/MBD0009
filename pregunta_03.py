# =============================================================================
# PREGUNTA 3: ESTRATEGIA DE RETENCION Y PREDICCION DE CHURN
# Modelo de clasificacion para predecir la desercion de clientes
# =============================================================================


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Carga y exploracion de datos
# -----------------------------------------------------------------------------
print("=" * 70)
print("PARTE 3: ESTRATEGIA DE RETENCION Y PREDICCION DE CHURN")
print("=" * 70)

# Cargamos el dataset de churn
datos_churn = pd.read_csv("data/data_churn.csv")

print("\n[DATOS] Exploracion inicial del dataset:")
print(f"   - Total de registros: {len(datos_churn):,}")
print(f"   - Variables disponibles: {datos_churn.columns.tolist()}")

# Analizamos la distribucion de la variable objetivo
tasa_churn = datos_churn['churn_real'].mean() * 100
print(f"\n[INFO] Distribucion de Churn:")
print(f"   - Clientes que abandonaron (churn=1): {datos_churn['churn_real'].sum():,} ({tasa_churn:.1f}%)")
print(f"   - Clientes retenidos (churn=0): {(datos_churn['churn_real'] == 0).sum():,} ({100-tasa_churn:.1f}%)")

# Estadisticas descriptivas
print("\n[STATS] Estadisticas descriptivas de las variables:")
print(datos_churn.describe().round(2).to_string())

# -----------------------------------------------------------------------------
# PREGUNTA 3.1: Entrenamiento del modelo de clasificacion
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PREGUNTA 3.1: MODELO DE CLASIFICACION PARA PREDECIR CHURN")
print("=" * 70)

# Preparación de las variables
# X: (caracteristicas del cliente)
# y: (churn)
variables_predictoras = ['antiguedad', 'gasto', 'soporte', 'satisfaccion']
X = datos_churn[variables_predictoras]
y = datos_churn['churn_real']

# Datos para en entrenamiento (70%) y prueba (30%)
# Usamos estratificacion para mantener la proporcion de churn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.30, 
    random_state=42,
    stratify=y
)

print(f"\n[DATOS] Division de datos:")
print(f"   - Conjunto de entrenamiento: {len(X_train):,} registros")
print(f"   - Conjunto de prueba: {len(X_test):,} registros")

# Normalizamos las variables para mejorar el rendimiento de algunos modelos
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

# Entrenamos varios modelos para comparar
print("\n[PROCESO] Entrenando modelos de clasificacion...")

# Modelo 1: Regresion Logistica (modelo base, interpretable)
modelo_logistico = LogisticRegression(random_state=42, max_iter=1000)
modelo_logistico.fit(X_train_escalado, y_train)

# Modelo 2: Random Forest (modelo de ensamble robusto)
modelo_rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
modelo_rf.fit(X_train, y_train)  # RF no requiere escalado

# Modelo 3: Gradient Boosting (alto rendimiento)
modelo_gb = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
modelo_gb.fit(X_train, y_train)

# Evaluacion de modelos
print("\n[RESULTADOS] Rendimiento de los modelos:")
print("-" * 70)

modelos = {
    'Regresion Logistica': (modelo_logistico, X_test_escalado),
    'Random Forest': (modelo_rf, X_test),
    'Gradient Boosting': (modelo_gb, X_test)
}

resultados_modelos = []

for nombre, (modelo, X_eval) in modelos.items():
    y_pred = modelo.predict(X_eval)
    y_proba = modelo.predict_proba(X_eval)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    
    resultados_modelos.append({
        'Modelo': nombre,
        'AUC-ROC': auc,
        'Predicciones': y_pred,
        'Probabilidades': y_proba
    })
    
    print(f"\n   * {nombre}:")
    print(f"     AUC-ROC: {auc:.4f}")

# Seleccionamos el mejor modelo basado en AUC-ROC
mejor_modelo_info = max(resultados_modelos, key=lambda x: x['AUC-ROC'])
print(f"\n[OK] Mejor modelo seleccionado: {mejor_modelo_info['Modelo']} (AUC: {mejor_modelo_info['AUC-ROC']:.4f})")

# Reporte de clasificacion del mejor modelo
print(f"\n[REPORTE] Clasificacion ({mejor_modelo_info['Modelo']}):")
print("-" * 70)
# Encontramos las predicciones del mejor modelo
for m in modelos:
    if m == mejor_modelo_info['Modelo']:
        y_pred_mejor = modelos[m][0].predict(modelos[m][1])
        break

print(classification_report(y_test, y_pred_mejor, 
                           target_names=['Retenido', 'Churn']))

# Matriz de confusion
print("[INFO] Matriz de Confusion:")
print("-" * 40)
cm = confusion_matrix(y_test, y_pred_mejor)
print(f"   Verdaderos Negativos (Retenidos predichos correctamente): {cm[0,0]:,}")
print(f"   Falsos Positivos (Retenidos predichos como churn): {cm[0,1]:,}")
print(f"   Falsos Negativos (Churn predichos como retenidos): {cm[1,0]:,}")
print(f"   Verdaderos Positivos (Churn predichos correctamente): {cm[1,1]:,}")

# Importancia de variables (para el modelo Random Forest o GB)
print("\n[IMPORTANTE] Importancia de las Variables (Random Forest):")
print("-" * 70)
importancias = pd.DataFrame({
    'Variable': variables_predictoras,
    'Importancia': modelo_rf.feature_importances_
}).sort_values('Importancia', ascending=False)

for _, row in importancias.iterrows():
    barra = "#" * int(row['Importancia'] * 50)
    print(f"   {row['Variable']:15} {row['Importancia']:.4f} {barra}")

# -----------------------------------------------------------------------------
# PREGUNTA 3.2: Realiza un análisis de lift y comenta los resultados.
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PREGUNTA 3.2: ANALISIS DE LIFT")
print("=" * 70)

# Usamos las probabilidades del mejor modelo
y_proba_mejor = mejor_modelo_info['Probabilidades']

# Creamos un DataFrame para el analisis de lift
analisis_lift = pd.DataFrame({
    'probabilidad_churn': y_proba_mejor,
    'churn_real': y_test.values
})

# Ordenamos por probabilidad descendente (clientes con mayor riesgo primero)
analisis_lift = analisis_lift.sort_values('probabilidad_churn', ascending=False)
analisis_lift['ranking'] = range(1, len(analisis_lift) + 1)

# Calculamos deciles (10 grupos de igual tamano)
analisis_lift['decil'] = pd.qcut(
    analisis_lift['ranking'], 
    q=10, 
    labels=['1 (Mayor riesgo)', '2', '3', '4', '5', '6', '7', '8', '9', '10 (Menor riesgo)']
)

# Calculamos metricas por decil
tabla_lift = analisis_lift.groupby('decil', observed=True).agg({
    'churn_real': ['count', 'sum', 'mean'],
    'probabilidad_churn': 'mean'
}).round(4)

tabla_lift.columns = ['Clientes', 'Churners', 'Tasa Churn', 'Prob Media']

# Calculamos el lift
tasa_base = y_test.mean()
tabla_lift['Lift'] = tabla_lift['Tasa Churn'] / tasa_base

# Calculamos churn acumulado
tabla_lift['Churners Acumulados'] = tabla_lift['Churners'].cumsum()
tabla_lift['% Churners Acumulados'] = (
    tabla_lift['Churners Acumulados'] / tabla_lift['Churners'].sum() * 100
)

print("\n[TABLA] Lift por Deciles:")
print("-" * 90)
print(tabla_lift.to_string())

# Analisis del lift
print("\n[ANALISIS] INTERPRETACION DEL ANALISIS DE LIFT:")
print("-" * 70)

lift_decil_1 = tabla_lift.loc['1 (Mayor riesgo)', 'Lift']
pct_churners_top20 = tabla_lift.iloc[:2]['% Churners Acumulados'].max()

print(f"""
El analisis de lift nos permite evaluar la capacidad del modelo para
concentrar a los clientes mas propensos a abandonar en los primeros deciles:

   RESULTADOS CLAVE:
   
   * Lift del primer decil: {lift_decil_1:.2f}x
     Esto significa que el 10% de clientes con mayor probabilidad predicha
     tiene {lift_decil_1:.1f} veces mas probabilidad de hacer churn que un
     cliente seleccionado aleatoriamente.
   
   * Cobertura del 20% superior: {pct_churners_top20:.1f}%
     Contactando solo al 20% de los clientes (los de mayor riesgo predicho),
     podemos alcanzar al {pct_churners_top20:.1f}% de todos los churners.

   IMPLICACIONES PARA CAMPAÑAS DE RETENCION:
   
   1. EFICIENCIA EN RECURSOS: En lugar de contactar a todos los clientes,
      podemos enfocar los esfuerzos de retencion en los primeros deciles
      donde el lift es mayor.
   
   2. ROI DE LA CAMPANA: Si el costo de perder un cliente es $X y el costo
      de la intervencion es $Y, el modelo es rentable cuando:
      Lift x Tasa de Exito de Retencion x $X > $Y
   
   3. PRIORIZACION: Los clientes en el decil 1 deberian recibir las
      intervenciones mas intensivas (llamadas personalizadas, ofertas
      exclusivas), mientras que los deciles intermedios pueden recibir
      intervenciones de menor costo (emails, descuentos automaticos).

   VARIABLES MAS INFLUYENTES EN CHURN:
""")

for _, row in importancias.iterrows():
    print(f"      - {row['Variable']}: {row['Importancia']*100:.1f}% de importancia")

print("""
   Estas variables nos indican donde enfocar las estrategias de retencion:
   - Mejorar la satisfaccion del cliente
   - Reducir la necesidad de llamadas a soporte
   - Incentivos especiales para clientes nuevos
""")

print("\n" + "=" * 70)
print("[OK] Analisis de Churn completado exitosamente")
print("=" * 70)
