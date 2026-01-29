# Problem Set 02: Marketing y Analítica del Retail
**Magíster in Business Analytics and Data Science - UDP**

Este repositorio contiene la resolución del Problem Set 02, enfocado en el procesamiento de datos no estructurados y sistemas de recomendación aplicados al retail. El proyecto aborda desafíos reales como el ruido en los datos, la dispersión (sparsity) y la evaluación de modelos de lenguaje.

## 📋 Índice
1. [Parte 1: NLP y Embeddings](#parte-1-nlp-y-embeddings)
2. [Parte 2: Modelado de Tópicos (BERTopic)](#parte-2-modelado-de-tópicos)
3. [Parte 3: Recomendación (Feedback Explícito)](#parte-3-recomendación-explícita)
4. [Parte 4: Recomendación (Feedback Implícito)](#parte-4-recomendación-implícita)
5. [Requisitos Técnicos](#requisitos-técnicos)

---

## 🧠 Parte 1: NLP y Embeddings
Utilizando el dataset `retail_reviews.csv`, se exploró la representación vectorial del lenguaje y la clasificación de sentimientos.

### 1.1. Análisis Semántico (Word2Vec)
Se entrenó un modelo para identificar términos similares. Resultados obtenidos:
* **Términos similares a "rápido":** *recomendado, excelente, eficaz, entrega, puntual.*
* **Similitud:** Se observaron puntajes de similitud superiores al **0.98**, validando que el modelo captura correctamente el contexto de eficiencia logística.
* **Interpretación Matemática:** Se utiliza la **Similitud Coseno** porque mide el ángulo entre vectores, ignorando su magnitud. Esto es crítico en retail, ya que palabras frecuentes (magnitud alta) no necesariamente son más relevantes semánticamente que palabras técnicas menos frecuentes.

### 1.2. Clasificación con Transformers (BERT vs Baseline)
Se comparó el rendimiento de clasificación binaria (Sentimiento Positivo/Negativo):
* **TF-IDF + Logistic Regression:** F1-Score de **0.9000**
* **BERT Embeddings + Logistic Regression:** F1-Score de **0.8924**
> **Nota:** Aunque TF-IDF tuvo un puntaje ligeramente superior, BERT demostró mayor capacidad para entender reseñas con sarcasmo o ambigüedad estructural.

---

## 📊 Parte 2: Modelado de Tópicos (BERTopic)
Se implementó **BERTopic** para descubrir temas latentes en las reseñas de los clientes sin supervisión previa.

* **Tópicos Identificados:** * **Tópico 0:** Satisfacción general y buen empaque (ej. "llegó", "bien", "empaquetado").
    * **Tópico 7:** Problemas con empaques defectuosos o sucios.
    * **Tópico 11:** Experiencias negativas con atención al cliente.
* **Visualización:** Se generaron mapas de distancia inter-tópica para analizar la jerarquía de los comentarios mediante UMAP y HDBSCAN.



---

## 🎬 Parte 3: Recomendación (Feedback Explícito)
Análisis del dataset `video_ratings.csv` para la predicción de valoraciones (1-5 estrellas).

* **Enfoque:** Filtrado Colaborativo basado en modelos.
* **Desafío:** El sistema maneja una **Sparsity Extrema** (20,000 registros para una matriz de 600 usuarios y 150 películas), optimizando la predicción de ratings para usuarios con pocos datos históricos.

---

## 🎵 Parte 4: Recomendación (Feedback Implícito)
Análisis de comportamiento mediante `music_logs.csv`, utilizando el conteo de reproducciones (`play_count`) como señal de interés.

* **Métrica de Evaluación:** Implementación de **NDCG@K** (Normalized Discounted Cumulative Gain).
* **Ejemplo de Resultado:** Para un ranking de 5 canciones donde los aciertos están en las posiciones 1, 3 y 4, el modelo calcula:
    * **DCG@5:** $\sum_{i=1}^{5} \frac{rel_i}{\log_2(i + 1)}$
    * **Resultado:** Permite penalizar si las canciones favoritas del usuario aparecen al final de la lista recomendada.



---

## 🛠️ Requisitos Técnicos
Para ejecutar el notebook `Tarea2_v1.ipynb`, asegúrese de contar con el siguiente entorno:

### Dependencias Principales
```bash
pip install tensorflow==2.16.2 tf_keras transformers sentence-transformers gensim scikit-learn bertopic plotly