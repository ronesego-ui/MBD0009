# =============================================================================
# PREGUNTA 5: INFERENCIA CAUSAL
# Estimacion del efecto de tratamiento publicitario (CATE)
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PARTE 5: INFERENCIA CAUSAL - EFECTO DE LA PUBLICIDAD")
print("=" * 70)

datos_causal = pd.read_csv("data/data_inferencia_causal.csv")
print(f"\n[DATOS] Observaciones: {len(datos_causal):,}")
print(datos_causal.describe().round(2).to_string())

n_tratados = datos_causal['W'].sum()
print(f"\n[INFO] Tratamiento (publicidad):")
print(f"   - Expuestos (W=1): {n_tratados:,} ({n_tratados/len(datos_causal)*100:.1f}%)")
print(f"   - Control (W=0): {(datos_causal['W'] == 0).sum():,}")

# S-LEARNER
print("\n" + "=" * 70)
print("PREGUNTA 5.1 - PARTE A: S-LEARNER")
print("=" * 70)

X = datos_causal[['edad', 'ingreso', 'W']].copy()
y = datos_causal['Y'].copy()
X_features = datos_causal[['edad', 'ingreso']].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_features_test = X_test[['edad', 'ingreso']].copy()

modelo_s = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
modelo_s.fit(X_train, y_train)

X_test_tratado = X_features_test.copy()
X_test_tratado['W'] = 1
X_test_control = X_features_test.copy()
X_test_control['W'] = 0

cate_s = modelo_s.predict(X_test_tratado) - modelo_s.predict(X_test_control)

resultados = pd.DataFrame({
    'edad': X_features_test['edad'].values,
    'ingreso': X_features_test['ingreso'].values,
    'cate_s': cate_s,
    'efecto_real': datos_causal.loc[X_test.index, 'efecto_real'].values
})

mae_s = mean_absolute_error(resultados['efecto_real'], cate_s)
print(f"\n   CATE promedio (S-Learner): ${cate_s.mean():.2f}")
print(f"   MAE: ${mae_s:.2f}")

# T-LEARNER
print("\n" + "=" * 70)
print("PREGUNTA 5.1 - PARTE B: T-LEARNER")
print("=" * 70)

datos_tratados = datos_causal[datos_causal['W'] == 1]
datos_control = datos_causal[datos_causal['W'] == 0]

modelo_t1 = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
modelo_t1.fit(datos_tratados[['edad', 'ingreso']], datos_tratados['Y'])

modelo_t0 = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
modelo_t0.fit(datos_control[['edad', 'ingreso']], datos_control['Y'])

cate_t = modelo_t1.predict(X_features_test) - modelo_t0.predict(X_features_test)
resultados['cate_t'] = cate_t

mae_t = mean_absolute_error(resultados['efecto_real'], cate_t)
print(f"\n   CATE promedio (T-Learner): ${cate_t.mean():.2f}")
print(f"   MAE: ${mae_t:.2f}")

# ANALISIS POR SEGMENTOS
print("\n" + "=" * 70)
print("ANALISIS DE CATE POR SEGMENTOS")
print("=" * 70)

resultados['seg_ingreso'] = pd.cut(resultados['ingreso'], 
    bins=[0, 35000, 50000, 65000, 200000],
    labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'])

cate_segmento = resultados.groupby('seg_ingreso', observed=True)['cate_t'].mean()
print("\n[RESULTADOS] CATE por segmento de ingreso:")
print(cate_segmento.round(2).to_string())

corr = resultados['cate_t'].corr(resultados['ingreso'])
print(f"\n   Correlacion CATE-Ingreso: {corr:.4f}")

# PREGUNTA 5.2
print("\n" + "=" * 70)
print("PREGUNTA 5.2: DECISION ESTRATEGICA DE MARKETING")
print("=" * 70)

if corr < 0:
    print("""
[ALERTA] CORRELACION NEGATIVA DETECTADA

La publicidad tiene MAYOR efecto en clientes de MENOR ingreso.

RECOMENDACIONES ESTRATEGICAS:

1. REORIENTAR EL TARGETING: Enfocar presupuesto publicitario en
   segmentos de ingreso bajo a medio donde el efecto es mayor.

2. SEGMENTAR PRESUPUESTO: Distribuir inversion segun el CATE
   estimado por segmento para maximizar el retorno.

3. REDUCIR GASTO EN SEGMENTOS DE BAJO RETORNO: Para clientes de
   alto ingreso, considerar canales organicos en lugar de pagados.

4. VALIDAR CON PRUEBAS A/B: Antes de escalar, realizar experimentos
   controlados en los segmentos identificados.
""")
else:
    print("\n   La correlacion es positiva. Mantener targeting en segmentos premium.")

print("\n" + "=" * 70)
print("[OK] Analisis de Inferencia Causal completado")
print("=" * 70)
