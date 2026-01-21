# Problem Set 01 - MBD0008

## Descripción General

Este repositorio contiene las soluciones al Problem Set 01 del curso de Magister en Business Data Science. Se abordan problemas de analisis de datos aplicados a decisiones de negocio en retail, retencion de clientes y marketing.

---

## Estructura del Proyecto

```text
Tarea01/
|-- main.py                     # Script principal para ejecutar todo
|-- pregunta_01.py              # Retail Analytics (Out-Of-Stock, GMROI, Markdown)
|-- pregunta_02.py              # Web Scraping (Portal Inmobiliario)
|-- pregunta_03.py              # Prediccion de Churn
|-- pregunta_04.py              # Customer Lifetime Value (CLTV)
|-- pregunta_05.py              # Inferencia Causal (CATE)
|-- requirements.txt            # Dependencias del proyecto
|-- README.md                   # Este archivo
+-- data/
    |-- maestro_productos.csv
    |-- inventario_diario.csv
    |-- transacciones_ventas.csv
    |-- data_churn.csv
    |-- data_rfm_cltv.csv
    +-- data_inferencia_causal.csv
```

---

## Instalación y Ejecución

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### Ejecución

Para ejecutar todas las preguntas:

```bash
python main.py
```

Para ejecutar una pregunta específica:

```bash
python pregunta_01.py  # Retail Analytics
python pregunta_02.py  # Web Scraping
python pregunta_03.py  # Churn
python pregunta_04.py  # CLTV
python pregunta_05.py  # Inferencia Causal
```

---

## Descripción de las Preguntas

### Parte 1: Calculo de KPIs (`pregunta_01.py`)

**Preguntas Respondidas:**

- 1.1: Cero Censurado (Out-Of-Stock) y Sustituciones de SKUs - Como afectan la integridad de los analisis historicos y medidas de ingenieria de datos recomendadas
- 1.2: Calculo de GMROI por categoria y explicacion de por que productos con bajo margen pueden tener alto GMROI
- 1.3: Calculo de Markdown por categoria e identificacion de problemas cuando GMROI alto coincide con Markdown > 20%

---

### Parte 2: Web Scraping (`pregunta_02.py`)

Implementacion de un scraper para obtener datos de casas y departamentos en Huechuraba desde Portal Inmobiliario.

**Metricas Calculadas:**

- Numero de propiedades scrapeadas (casas y departamentos)
- Precio promedio y mediana (UF)
- Precio promedio por m2 (UF/m2)

**Preguntas Respondidas:**

- 2.1: Cuadro comparativo de metricas y analisis de resultados
- 2.2: Desafios eticos y tecnicos (privacidad, terminos de servicio, bloqueos) y medidas recomendadas para produccion

---

### Parte 3: Predicción de Churn (`pregunta_03.py`)

Modelo de clasificación para identificar clientes en riesgo de abandonar.

**Modelos Implementados:**

- Regresión Logística
- Random Forest
- Gradient Boosting

**Análisis Incluidos:**

- 3.1: Entrenamiento y evaluación de modelos con métricas AUC-ROC, precisión y recall
- 3.2: Análisis de Lift por deciles para optimizar campañas de retención

**Variables Predictoras:**

- Antigüedad del cliente
- Gasto promedio
- Llamadas a soporte
- Nivel de satisfacción

---

### Parte 4: Customer Lifetime Value (`pregunta_04.py`)

Modelamiento del valor de vida del cliente.

**Modelos Implementados:**

- 4.1: Regresión Lineal tradicional
- 4.1: Modelo Gamma-Gamma probabilístico

**Análisis Incluidos:**

- Comparación de métodos
- 4.2: Explicación detallada de cómo el modelo Gamma-Gamma maneja outliers mediante shrinkage bayesiano

---

### Parte 5: Inferencia Causal (`pregunta_05.py`)

Estimación del efecto causal de la publicidad en las ventas.

**Meta-Learners Implementados:**

- **S-Learner:** Modelo único con tratamiento como variable
- **T-Learner:** Modelos separados para tratados y control

**Análisis Incluidos:**

- 5.1: Estimación del CATE (Conditional Average Treatment Effect) por segmentos
- 5.2: Recomendación estratégica sobre asignación presupuestaria basada en la correlación del efecto con el ingreso

---

## Datasets Utilizados

| Archivo | Descripción | Registros |
|---------|-------------|-----------|
| `maestro_productos.csv` | Catálogo de productos con costos y precios | ~1,000 |
| `inventario_diario.csv` | Snapshot diario de inventario | ~365,000 |
| `transacciones_ventas.csv` | Transacciones de ventas | ~100,000 |
| `data_churn.csv` | Datos de churn de clientes | ~100,000 |
| `data_rfm_cltv.csv` | Datos RFM para CLTV | ~100,000 |
| `data_inferencia_causal.csv` | Datos experimentales para causalidad | ~5,000 |

---

## Supuestos y Consideraciones

1. **Pregunta 1:** Se asume que el inventario promedio refleja la inversión típica en stock durante el período analizado.

2. **Pregunta 3:** Se utilizó un split 70/30 para entrenamiento/prueba con estratificación para mantener la proporción de churn.

3. **Pregunta 4:** El modelo Gamma-Gamma requiere que frequency > 0. Los clientes sin compras fueron excluidos del análisis.

4. **Pregunta 5:** Se asume que los datos contienen el "efecto real" como ground truth para validación de los métodos.

---

## Autor

Grupo 2 - MBD0009
Daniel Avello
Felipe Valdivia
Roberto Sepulveda

---
"# MBD0009" 
