# Problem Set 02: Marketing y Analítica del Retail
**Magíster in Business Analytics and Data Science - UDP**

Este repositorio contiene la resolución integral del Problem Set 02, enfocado en el procesamiento de datos no estructurados y el desarrollo de sistemas de recomendación. El proyecto enfrenta desafíos reales como ruido en los datos, textos sucios y matrices de consumo con alta dispersión (*sparsity*).

## 📋 Índice
1. [Limpieza y Normalización de Datos](#0-limpieza-y-normalización-de-datos)
2. [Parte 1: NLP y Embeddings](#parte-1-nlp-y-embeddings)
3. [Parte 2: Modelado de Tópicos (BERTopic)](#parte-2-modelado-de-tópicos)
4. [Parte 3: Recomendación (Feedback Explícito)](#parte-3-recomendación-explícita)
5. [Parte 4: Recomendación (Feedback Implícito)](#parte-4-recomendación-implícita)
6. [Parte 5: Re-ranking y Estrategia de Negocio](#parte-5-re-ranking-y-estrategia-de-negocio)
7. [Requisitos Técnicos y Compatibilidad](#requisitos-técnicos-y-compatibilidad)

---

## 🧹 0. Limpieza y Normalización de Datos
Antes de cada análisis, se aplicó un pipeline de preprocesamiento específico para mitigar el "ruido intencional" de los datasets:

* **Normalización de Texto:** Conversión a minúsculas y eliminación de caracteres especiales, puntuación y números mediante expresiones regulares (`re`).
* **Tratamiento de Nulos:** Eliminación de filas con reseñas vacías para asegurar la integridad de la tokenización y el entrenamiento de modelos.

---

## 🧠 Parte 1: NLP y Embeddings
Se exploraron técnicas avanzadas para transformar texto en representaciones vectoriales:
* **Word2Vec:** Análisis de cercanía semántica entre conceptos de retail.
* **BERT vs TF-IDF:** Comparativa de modelos para clasificación de sentimiento. Se demostró que los embeddings de BERT capturan mejor el contexto en casos de reseñas complejas o sarcásticas.

---

## 📊 Parte 2: Modelado de Tópicos (BERTopic)
Se utilizó **BERTopic** para el descubrimiento automático de temas en las reseñas de los clientes.
* **Tópicos Clave:** Se lograron identificar clústeres específicos sobre satisfacción de empaque (Tópico 0), problemas de limpieza/estado de productos (Tópico 7) y deficiencias en el soporte post-venta (Tópico 11).

---

## 🎬 Parte 3: Recomendación (Feedback Explícito)
Implementación de un sistema basado en `video_ratings.csv`:
* **SVD (Singular Value Decomposition):** El modelo gestiona la alta dispersión de datos mediante Filtrado Colaborativo, permitiendo predecir el interés de un usuario por ítems que aún no ha calificado.

---

## 🎵 Parte 4: Recomendación (Feedback Implícito)
Análisis de preferencias musicales mediante logs de consumo.
* **Métrica NDCG@K:** Se utilizó para evaluar la calidad del ranking, penalizando las recomendaciones relevantes que aparecen muy abajo en la lista.
* **Fórmula aplicada:** $$DCG_k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i + 1)}$$

---

## 🚀 Parte 5: Re-ranking y Estrategia de Negocio
Se implementó un algoritmo de post-procesamiento para alinear las predicciones del modelo con los objetivos comerciales de la empresa.
* **Optimización de Margen:** Se ajustaron los scores originales de recomendación multiplicándolos por factores basados en la rentabilidad del producto.
    * **High Margin:** Multiplicador de **1.2** (+20% de visibilidad).
    * **Low Margin:** Multiplicador de **0.9** (-10% de visibilidad).
* **Resultado:** El sistema no solo recomienda lo que al usuario le gusta, sino que prioriza aquellos productos que generan mayor valor para el negocio sin perder la relevancia personal.

---

## 🛠️ Requisitos Técnicos y Compatibilidad

### ⚠️ Configuración de Compatibilidad (Crítico)
Para asegurar el funcionamiento del código en entornos con **Python 3.14+** y **TensorFlow 2.16+**, se configuró el motor heredado de Keras de la siguiente manera:

```python
import os
# Forzamos el uso del motor antiguo de Keras 2
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tf_keras as keras


### 📦 Librerías a Instalar
Para preparar el entorno, ejecuta los siguientes comandos en tu terminal:

# Procesamiento de datos y visualización
pip install pandas numpy matplotlib seaborn scikit-learn

# Deep Learning y Modelos de Lenguaje (NLP)
pip install tensorflow tf_keras transformers sentence-transformers

# Tópicos y Recomendación
pip install bertopic nltk surprise