import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB

# ==========================================
# i. Cargar conjunto de documentos
# ==========================================
documentos = [
    "Manual de configuración de servidores y redes en la nube AWS",
    "Políticas corporativas de recursos humanos, vacaciones y beneficios",
    "Código fuente para integración de API REST en Python y Django",
    "Guía de onboarding y capacitación para nuevos empleados",
    "Reporte de incidente y post-mortem de caída de base de datos SQL",
    "Arquitectura de microservicios con Docker y Kubernetes",
    (
        "Guía técnica de infraestructura en la nube y despliegue de"
        " contenedores"
    ),
    "Manual para desplegar aplicaciones y servidores en Microsoft Azure",
    "Evaluación de desempeño y métricas del personal de recursos humanos",
]

etiquetas = [
    "Infraestructura",
    "Recursos Humanos",
    "Desarrollo",
    "Recursos Humanos",
    "Infraestructura",
    "Desarrollo",
    "Infraestructura",
    "Infraestructura",
    "Recursos Humanos",
]

# ==========================================
# ii. Vectorizar el texto utilizando TF-IDF
# ==========================================
vectorizador = TfidfVectorizer(
    stop_words=["de", "y", "en", "para", "con", "la", "del"]
)
X_vectores = vectorizador.fit_transform(documentos)

# ==========================================
# iii. Entrenar modelo de clasificación (Naive Bayes)
# ==========================================
clasificador = MultinomialNB()
clasificador.fit(X_vectores, etiquetas)

# ==========================================
# iv. Clasificar documentos nuevos (Pruebas)
# ==========================================
print("--- PRUEBAS DE CLASIFICACIÓN AUTOMATIZADA ---")
docs_prueba = [
    "Guía técnica para desplegar contenedores en Microsoft Azure",
    "Protocolo de bienvenida y entrega de equipo para personal de nuevo ingreso",
]

vectores_prueba = vectorizador.transform(docs_prueba)
predicciones = clasificador.predict(vectores_prueba)

for doc, cat in zip(docs_prueba, predicciones):
  print(f"Documento: '{doc}'\n -> Categoría asignada: {cat}\n")


# ==========================================
# 5. Implementación de Búsqueda Semántica
# ==========================================
def buscar_documentos(consulta):
  print(f"Consulta: '{consulta}'")
  vector_consulta = vectorizador.transform([consulta])
  similitudes = cosine_similarity(vector_consulta, X_vectores).flatten()
  indices_ordenados = similitudes.argsort()[::-1]

  for idx in indices_ordenados:
    if similitudes[idx] > 0:
      print(
          f"  [Similitud: {similitudes[idx]:.4f}] {documentos[idx]} (Categoría:"
          f" {etiquetas[idx]})"
      )
  print("-" * 60)


print("--- PRUEBAS DE BÚSQUEDA SEMÁNTICA ---")
buscar_documentos("configurar bases de datos y servidores")
buscar_documentos("capacitación y beneficios para nuevos empleados")
