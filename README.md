# Problem Set 02: Marketing y Analítica del Retail
**Magíster in Business Analytics and Data Science - UDP**

Este repositorio contiene la resolución integral del Problem Set 02, enfocado en el procesamiento de datos no estructurados y el desarrollo de sistemas de recomendación. El proyecto simula un entorno real de retail enfrentando desafíos como ruido en los datos, textos sin procesar y matrices de consumo dispersas (sparsity).

## 📋 Índice
1. [Parte 1: NLP y Embeddings](#parte-1-nlp-y-embeddings)
2. [Parte 2: Modelado de Tópicos (BERTopic)](#parte-2-modelado-de-tópicos)
3. [Parte 4: Recomendación (Feedback Explícito)](#parte-3-recomendación-explícita)
4. [Parte 4: Recomendación (Feedback Implícito)](#parte-4-recomendación-implícita)
5. [Requisitos Técnicos y Compatibilidad](#requisitos-técnicos-y-compatibilidad)

---

## 🧠 Parte 1: NLP y Embeddings
Utilizando el dataset `retail_reviews.csv`, se exploró la representación vectorial del lenguaje y la clasificación de sentimientos.

### 1.1. Análisis Semántico (Word2Vec)
Se entrenó un modelo Word2Vec sobre el corpus de reseñas para capturar la semántica del negocio:
* **Similitud Semántica:** Para el término **"rápido"**, el modelo identificó términos como *recomendado, excelente, eficaz, entrega y puntual* con similitudes de hasta **0.99**.
* **Interpretación Matemática:** Se implementó la **Similitud Coseno** en lugar del Producto Punto. Esto es fundamental en NLP porque la Similitud Coseno normaliza los vectores por su norma $L2$, permitiendo medir la cercanía en ángulo y no por la magnitud (frecuencia) de las palabras.
* **Álgebra Vectorial:** Se validó la coherencia del espacio latente mediante analogías, permitiendo entender relaciones entre atributos de productos y sentimientos.

### 1.2. Clasificación de Sentimientos (Transformers vs Baseline)
Se comparó un enfoque estadístico tradicional contra un modelo de lenguaje avanzado:
* **Baseline (TF-IDF + LogReg):** Logró un F1-Score de **0.9000**.
* **BERT (paraphrase-multilingual-MiniLM-L12-v2):** Logró un F1-Score de **0.8924**.
* **Conclusión:** Aunque los puntajes son cercanos, el modelo basado en BERT demuestra una mejor capacidad de generalización ante estructuras lingüísticas complejas y sarcasmo.

---

## 📊 Parte 2: Modelado de Tópicos (BERTopic)
Se utilizó la arquitectura **BERTopic** para el descubrimiento automático de temas en las reseñas.

* **Flujo Técnico:** Generación de Embeddings → Reducción de dimensionalidad (UMAP) → Clustering (HDBSCAN) → c-TF-IDF para la extracción de palabras clave.
* **Tópicos Clave Identificados:**
    * **Tópico 0:** Satisfacción general y logística (palabras: llegó, bien, rápido).
    * **Tópico 7:** Problemas críticos con productos defectuosos o dañados.
    * **Tópico 11:** Malas experiencias con el servicio de soporte y atención.



---

## 🎬 Parte 3: Recomendación (Feedback Explícito)
Basado en `video_ratings.csv`, se desarrolló un sistema para predecir la valoración (1 a 5) de películas.

* **Desafío de Sparsity:** Con ~20,000 registros para 600 usuarios y 150 películas, el modelo utiliza técnicas de **Filtrado Colaborativo** para predecir el interés de un usuario en ítems que nunca ha consumido, optimizando la oferta de contenido.

---

## 🎵 Parte 4: Recomendación (Feedback Implícito)
Análisis de `music_logs.csv` utilizando el conteo de reproducciones como métrica de interés.

* **Métrica de Evaluación:** Se implementó la métrica **NDCG@K** (Normalized Discounted Cumulative Gain).
* **Lógica de Ranking:** El modelo evalúa si las canciones con más "play counts" reales aparecen en los primeros lugares de la recomendación.
* **Fórmula aplicada:** $$DCG_k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i + 1)}$$
  Donde $rel_i$ es la relevancia del ítem en la posición $i$. El resultado final se normaliza (NDCG) para comparar entre diferentes usuarios.

---

## 🛠️ Requisitos Técnicos y Compatibilidad

### ⚠️ Configuración de Compatibilidad (Crítico)
Debido a que el entorno utiliza un Kernel de **Python 3.14.2** y **TensorFlow 2.16.2**, es estrictamente necesario forzar el uso de Keras Legacy para asegurar la estabilidad de los modelos:

```python
pip install tensorflow==2.16.2 tf_keras transformers sentence-transformers gensim scikit-learn bertopic plotly
import os
# Forzamos a TensorFlow a usar el motor antiguo de Keras 2
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
print(f"Versión de TF detectada: {tf.__version__}")

