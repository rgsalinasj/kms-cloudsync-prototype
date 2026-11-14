# Prototipo de Sistema de Gestión del Conocimiento (KMS) - CloudSync Engineering

**Autor:** Rafael Guadalupe Salinas Jiménez  
**Materia:** Sistemas y Tecnologías de Administración del Conocimiento  

## Descripción del Proyecto
Este repositorio contiene el prototipo funcional de un Sistema de Gestión del Conocimiento desarrollado para la empresa ficticia **CloudSync Engineering** (consultoría tecnológica especializada en Cloud y Machine Learning). El sistema automatiza la organización y recuperación de documentos técnicos y administrativos mediante técnicas de Procesamiento de Lenguaje Natural (NLP).

## Funcionalidades Implementadas
1. **Vectorización de Texto (TF-IDF):** Transforma los documentos corporativos en representaciones numéricas ponderadas filtrando palabras vacías (*stop words*).
2. **Clasificación Automatizada:** Entrena un modelo **Multinomial Naive Bayes** para categorizar automáticamente nuevos documentos en tres áreas principales: *Infraestructura*, *Desarrollo* y *Recursos Humanos*.
3. **Búsqueda Semántica:** Procesa consultas en lenguaje natural y calcula la **similitud del coseno** contra la base documental para recuperar los archivos más relevantes ordenados por puntuación.

## Estructura del Repositorio
* `prototipo_kms.py`: Script principal en Python con el flujo completo de vectorización, entrenamiento, clasificación y búsqueda semántica.
* `documentos_prueba.txt`: Conjunto de documentos corporativos, casos de prueba y consultas utilizadas para validar el prototipo.
* `README.md`: Documentación general del proyecto e instrucciones de ejecución.

## Requisitos y Ejecución
Para ejecutar el prototipo localmente se requiere **Python 3.8+** y la librería `scikit-learn`:

```bash
pip install scikit-learn numpy pandas
python prototipo_kms.py
