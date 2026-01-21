# =============================================================================
# ARCHIVO PRINCIPAL - PROBLEM SET 01
# Ejecuta todas las preguntas en secuencia
# =============================================================================

import subprocess
import sys
import os

# Cambiamos al directorio del script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("       PROBLEM SET 01 - MBD0009")
print("       Analisis de Datos para la Toma de Decisiones")
print("=" * 70)
print("\nEjecutando todos los analisis (Preguntas 1, 2, 3, 4 y 5)...\n")

# Lista de scripts a ejecutar
scripts = [
    ("pregunta_01.py", "PARTE 1: Retail Analytics"),
    ("pregunta_02.py", "PARTE 2: Web Scraping"),
    ("pregunta_03.py", "PARTE 3: Prediccion de Churn"),
    ("pregunta_04.py", "PARTE 4: Customer Lifetime Value"),
    ("pregunta_05.py", "PARTE 5: Inferencia Causal"),
]

# Ejecutamos cada script
for script, descripcion in scripts:
    print("\n" + "=" * 70)
    print(f"EJECUTANDO: {descripcion}")
    print("=" * 70)
    
    try:
        resultado = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            text=True
        )
        if resultado.returncode != 0:
            print(f"\n[AVISO] {script} termino con codigo {resultado.returncode}")
    except Exception as e:
        print(f"\n[ERROR] Error ejecutando {script}: {e}")
        
    print("\n" + "-" * 70)

print("\n" + "=" * 70)
print("       [OK] ANALISIS COMPLETO FINALIZADO")
print("=" * 70)
