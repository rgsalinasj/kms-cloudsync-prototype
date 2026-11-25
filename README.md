# Sistema de Gestión del Conocimiento Impulsado por IA — CloudSync Engineering

- **Autor:** Rafael Guadalupe Salinas Jiménez (AL03085966)
- **Materia:** Sistemas y Tecnologías de Administración del Conocimiento
- **Prototipo Desplegado:** https://kms-cloudsync-prototype-gegukdpix7j9dxz8nzr3ka.streamlit.app/
- **Repositorio:** https://github.com/rgsalinasj/kms-cloudsync-prototype

---

## 1. Descripción General y Contexto Organizacional
**CloudSync Engineering** es una firma de consultoría tecnológica especializada en arquitecturas Cloud (AWS/Azure), desarrollo de software nativo en la nube, soluciones de Machine Learning y soporte de infraestructura. El sistema resuelve la fragmentación de conocimiento técnico y administrativo centralizando manuales de configuración, repositorios de código, reportes post-mortem y guías de recursos humanos en una plataforma unificada asistida por Inteligencia Artificial.

## 2. Arquitectura del Sistema
El sistema sigue una arquitectura modular de tres capas ejecutada sobre Python y Streamlit:

```text
+-----------------------------------------------------------------------+
|                    INTERFAZ WEB INTEGRADA (Streamlit)                 |
|  [1. Clasificador]  [2. Recomendador]  [3. Chatbot]  [4. Grafo]       |
+-----------------------------------+-----------------------------------+
                                    | Intercambio de estado (session_state)
+-----------------------------------v-----------------------------------+
|               CAPA DE PROCESAMIENTO SEMÁNTICO E IA                    |
|  • Vectorizador Compartido: TfidfVectorizer (n-gramas 1,2 + stopwords)|
|  • Clasificador: Multinomial Naive Bayes (alpha=0.5)                  |
|  • Motor de Similitud: Similitud del Coseno (Documento e Historial)   |
|  • Motor de Grafo: NetworkX (DiGraph) + PyVis (Renderizado HTML)      |
+-----------------------------------+-----------------------------------+
                                    |
+-----------------------------------v-----------------------------------+
|                 REPOSITORIO DE CONOCIMIENTO (Docs)                    |
|  Metadatos estructurados: ID, Título, Tipo Ontológico, Dominio,       |
|  Conceptos Clave y Contenido Textual (CS-01 a CS-09+)                 |
+-----------------------------------------------------------------------+
