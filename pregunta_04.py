# =============================================================================
# PREGUNTA 4: CUSTOMER LIFETIME VALUE (CLTV)
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    from lifetimes import GammaGammaFitter
    LIFETIMES_DISPONIBLE = True
except ImportError:
    LIFETIMES_DISPONIBLE = False

print("=" * 70)
print("PARTE 4: CUSTOMER LIFETIME VALUE (CLTV)")
print("=" * 70)

datos_cltv = pd.read_csv("data/data_rfm_cltv.csv")
print(f"\n[DATOS] Clientes: {len(datos_cltv):,}")
print(datos_cltv.describe().round(2).to_string())

# REGRESION LINEAL
print("\n" + "=" * 70)
print("PREGUNTA 4.1 - PARTE A: REGRESION LINEAL")
print("=" * 70)

X = datos_cltv[['frequency']].copy()
y = datos_cltv['monetary_value'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train)
y_pred_lineal = modelo_lineal.predict(X_test)

rmse_lineal = np.sqrt(mean_squared_error(y_test, y_pred_lineal))
mae_lineal = mean_absolute_error(y_test, y_pred_lineal)

print(f"\n   Intercepto: {modelo_lineal.intercept_:.4f}")
print(f"   Coeficiente: {modelo_lineal.coef_[0]:.4f}")
print(f"   RMSE: ${rmse_lineal:,.2f}")
print(f"   MAE: ${mae_lineal:,.2f}")

# GAMMA-GAMMA
print("\n" + "=" * 70)
print("PREGUNTA 4.1 - PARTE B: MODELO GAMMA-GAMMA")
print("=" * 70)

datos_gg = datos_cltv[datos_cltv['frequency'] > 0].copy()
mu = datos_gg['monetary_value'].mean()
var = datos_gg['monetary_value'].var()
alpha = (mu ** 2) / var
beta = var / mu

datos_gg['valor_esperado_gg'] = (alpha + datos_gg['frequency'] * datos_gg['monetary_value']) / (beta + datos_gg['frequency'])
factor_escala = mu / datos_gg['valor_esperado_gg'].mean()
datos_gg['valor_esperado_gg'] *= factor_escala

rmse_gg = np.sqrt(mean_squared_error(datos_gg['monetary_value'], datos_gg['valor_esperado_gg']))
mae_gg = mean_absolute_error(datos_gg['monetary_value'], datos_gg['valor_esperado_gg'])

print(f"\n   Shape (alpha): {alpha:.4f}")
print(f"   Scale (beta): {beta:.4f}")
print(f"   RMSE: ${rmse_gg:,.2f}")
print(f"   MAE: ${mae_gg:,.2f}")

# COMPARACION DE MODELOS
print("\n" + "=" * 70)
print("COMPARACION DE MODELOS")
print("=" * 70)
print(f"{'Metrica':<20} {'Lineal':>15} {'Gamma-Gamma':>15}")
print("-" * 50)
print(f"{'RMSE':<20} {'$'+f'{rmse_lineal:,.2f}':>15} {'$'+f'{rmse_gg:,.2f}':>15}")
print(f"{'MAE':<20} {'$'+f'{mae_lineal:,.2f}':>15} {'$'+f'{mae_gg:,.2f}':>15}")

# PREGUNTA 4.2
print("\n" + "=" * 70)
print("PREGUNTA 4.2: Explica cómo el modelo gamma-gamma trata a los " \
"clientes outliers y por qué este enfoque podría ser más robusto que el lineal.")
print("=" * 70)
print("""
COMO TRATA GAMMA-GAMMA A LOS OUTLIERS:

1. SHRINKAGE BAYESIANO: Las predicciones se "encogen" hacia la media
   poblacional. Clientes con pocas compras pero valor alto reciben
   predicciones conservadoras.

2. DISTRIBUCION GAMMA: Modela naturalmente datos positivos y sesgados,
   esperando valores extremos como parte de la variabilidad natural.

3. REGRESION LINEAL: Los outliers tienen influencia desproporcionada
   en los coeficientes. No hay mecanismo de moderacion.

4. GAMMA-GAMMA: Usa informacion de otros clientes para moderar
   predicciones individuales. Nunca predice valores negativos.

CONCLUSION: Gamma-Gamma es mas robusto porque respeta la naturaleza
de los datos de gasto y protege contra sobreestimacion de clientes
con historial limitado.
""")
print("[OK] Analisis de CLTV completado")
