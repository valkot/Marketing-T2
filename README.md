# Problem Set 02: Marketing y Analítica del Retail
**Magíster in Business Analytics and Data Science - UDP**

Este repositorio contiene la resolución integral del Problem Set 02, enfocado en el procesamiento de datos no estructurados y el desarrollo de sistemas de recomendación. El proyecto enfrenta desafíos reales como ruido en los datos, textos sucios y matrices de consumo con alta dispersión (sparsity).

## 📋 Índice
1. [Limpieza y Normalización de Datos](#0-limpieza-y-normalización-de-datos)
2. [Parte 1: NLP y Embeddings](#parte-1-nlp-y-embeddings)
3. [Parte 2: Modelado de Tópicos (BERTopic)](#parte-2-modelado-de-tópicos)
4. [Parte 3: Recomendación (Feedback Explícito)](#parte-3-recomendación-explícita)
5. [Parte 4: Recomendación (Feedback Implícito)](#parte-4-recomendación-implícita)
6. [Requisitos Técnicos y Compatibilidad](#requisitos-técnicos-y-compatibilidad)

---

## 🧹 0. Limpieza y Normalización de Datos
Antes de cada análisis, se aplicó un pipeline de preprocesamiento específico para mitigar el "ruido intencional" de los datasets:

* **Dataset de Reseñas (`retail_reviews.csv`):** * **Normalización de Texto:** Conversión a minúsculas y eliminación de caracteres especiales, puntuación y números mediante expresiones regulares (`re`).
    * **Tratamiento de Nulos:** Eliminación de filas con reseñas vacías para evitar errores en la tokenización.
* **Dataset de Videos (`video_ratings.csv`):** * **Validación de Rangos:** Limpieza de ratings fuera del umbral esperado (1-5).
    * **Estructuración:** Aseguramiento de tipos de datos enteros para los IDs de usuario y película para facilitar la construcción de la matriz de dispersión.
* **Dataset de Música (`music_logs.csv`):** * **Manejo de Outliers:** Identificación de `play_counts` anómalos que podrían sesgar el cálculo del NDCG.
    * **Agregación:** Consolidación de registros duplicados de interacciones usuario-canción.



---

## 🧠 Parte 1: NLP y Embeddings
### 1.1. Análisis Semántico (Word2Vec)
Se entrenó un modelo Word2Vec para capturar la semántica del negocio:
* **Similitud Semántica:** Para el término **"rápido"**, el modelo identificó términos como *recomendado, excelente, eficaz, entrega y puntual* con similitudes de hasta **0.99**.
* **Interpretación Matemática:** Se implementó la **Similitud Coseno** en lugar del Producto Punto. Esto es fundamental porque la Similitud Coseno normaliza los vectores por su norma $L2$, permitiendo medir la cercanía en ángulo y no por la frecuencia (magnitud) de las palabras, evitando que términos comunes dominen el espacio semántico.

### 1.2. Clasificación de Sentimientos (Transformers vs Baseline)
* **Baseline (TF-IDF + LogReg):** F1-Score de **0.9000**.
* **BERT (`paraphrase-multilingual-MiniLM-L12-v2`):** F1-Score de **0.8924**.
* **Conclusión:** BERT ofrece una comprensión contextual superior, identificando sentimientos negativos incluso cuando se usan palabras superficialmente positivas (ej. sarcasmo).

---

## 📊 Parte 2: Modelado de Tópicos (BERTopic)
Se utilizó **BERTopic** para el descubrimiento automático de temas en las reseñas.
* **Tópicos Clave:** Se aislaron temas sobre satisfacción de empaque (Tópico 0), productos dañados (Tópico 7) y deficiencias en el soporte al cliente (Tópico 11).

---

## 🎬 Parte 3: Recomendación (Feedback Explícito)
Basado en `video_ratings.csv`, se predice la valoración de películas.
* **Sparsity:** El modelo gestiona la alta dispersión de datos mediante Filtrado Colaborativo, prediciendo ratings para pares usuario-película inexistentes en el entrenamiento.

---

## 🎵 Parte 4: Recomendación (Feedback Implícito)
Análisis de `music_logs.csv` mediante el conteo de reproducciones.
* **Métrica NDCG@K:** Se evaluó la calidad del ranking para asegurar que los elementos más escuchados por el usuario aparezcan al inicio de su lista recomendada.
* **Fórmula:** $$DCG_k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i + 1)}$$



---

## 🛠️ Requisitos Técnicos y Compatibilidad

### ⚠️ Configuración de Compatibilidad (Crítico)
Debido a la arquitectura del entorno (Python 3.14.2 / TF 2.16.2), se debe forzar el motor heredado de Keras:

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
print(f"Versión de TF detectada: {tf.__version__}")

pip install tensorflow==2.16.2 tf_keras transformers sentence-transformers gensim scikit-learn bertopic plotly