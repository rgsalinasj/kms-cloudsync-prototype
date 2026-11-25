import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="CloudSync Engineering - KMS IA", page_icon="☁️", layout="wide")

# ==============================================================================
# 1. REPOSITORIO CENTRAL DE CONOCIMIENTO (Basado en Fase I - CloudSync Engineering)
# ==============================================================================
if "docs" not in st.session_state:
    st.session_state.docs = [
        {
            "id": "CS-01",
            "titulo": "Manual de configuración de servidores y redes en la nube AWS",
            "tipo": "Manual de Configuración",
            "categoria": "Infraestructura",
            "conceptos": ["AWS", "Servidores", "Redes VPC", "Terraform"],
            "contenido": "Estándar técnico para aprovisionar instancias EC2, subredes privadas VPC, balanceadores ALB y reglas de Security Groups en AWS mediante scripts de Terraform."
        },
        {
            "id": "CS-02",
            "titulo": "Políticas corporativas de recursos humanos, vacaciones y beneficios",
            "tipo": "Guía de Inducción",
            "categoria": "Recursos Humanos",
            "conceptos": ["Vacaciones", "Beneficios", "Políticas", "Seguridad de Información"],
            "contenido": "Lineamientos sobre solicitud de vacaciones anuales, esquema híbrido de trabajo, prestaciones médicas y cumplimiento de políticas de seguridad de la información."
        },
        {
            "id": "CS-03",
            "titulo": "Código fuente para integración de API REST en Python y Django",
            "tipo": "Repositorio de Código",
            "categoria": "Desarrollo",
            "conceptos": ["Python", "API REST", "Django", "Autenticación JWT"],
            "contenido": "Plantilla base en Python y Django REST Framework con autenticación JWT, paginación estándar, manejo de excepciones y pruebas unitarias automatizadas."
        },
        {
            "id": "CS-04",
            "titulo": "Guía de onboarding y capacitación para nuevos empleados",
            "tipo": "Guía de Inducción",
            "categoria": "Recursos Humanos",
            "conceptos": ["Onboarding", "Capacitación", "GitHub", "Slack"],
            "contenido": "Ruta de inducción de 2 semanas para nuevos ingenieros: configuración de accesos a GitHub, Confluence, canales de Slack y certificaciones cloud iniciales."
        },
        {
            "id": "CS-05",
            "titulo": "Reporte de incidente y post-mortem de caída de base de datos SQL",
            "tipo": "Reporte de Incidente",
            "categoria": "Infraestructura",
            "conceptos": ["Base de Datos SQL", "Post-mortem", "Alta Disponibilidad", "Backups"],
            "contenido": "Análisis de causa raíz por saturación de conexiones en PostgreSQL productivo. Acción correctiva: implementar connection pooling con PgBouncer y actualizar manual de configuración."
        },
        {
            "id": "CS-06",
            "titulo": "Arquitectura de microservicios con Docker y Kubernetes",
            "tipo": "Manual de Configuración",
            "categoria": "Desarrollo",
            "conceptos": ["Docker", "Kubernetes", "Microservicios", "CI/CD"],
            "contenido": "Especificación de contenedores Docker multi-stage, manifiestos de despliegue en clústeres Kubernetes (EKS/AKS) y pipelines de integración continua."
        },
        {
            "id": "CS-07",
            "titulo": "Manual para desplegar aplicaciones y servidores en Microsoft Azure",
            "tipo": "Manual de Configuración",
            "categoria": "Infraestructura",
            "conceptos": ["Microsoft Azure", "Servidores", "Despliegue", "Redes VPC"],
            "contenido": "Procedimiento paso a paso para desplegar aplicaciones en Azure App Service, configurar Azure Virtual Network y gestionar secretos en Key Vault."
        },
        {
            "id": "CS-08",
            "titulo": "Pipeline de entrenamiento e inferencia de modelos ML en producción",
            "tipo": "Repositorio de Código",
            "categoria": "Inteligencia Artificial",
            "conceptos": ["Machine Learning", "Python", "MLOps", "API REST"],
            "contenido": "Arquitectura MLOps para entrenamiento automatizado, versionado de modelos con MLflow y exposición de endpoints de inferencia mediante contenedores Docker."
        },
        {
            "id": "CS-09",
            "titulo": "Evaluación de desempeño y métricas del personal técnico",
            "tipo": "Guía de Inducción",
            "categoria": "Recursos Humanos",
            "conceptos": ["Evaluación", "OKRs", "Tech Talks", "Capacitación"],
            "contenido": "Matriz trimestral de evaluación basada en calidad de Pull Requests, participación en Tech Talks quincenales, certificaciones cloud y entrega de proyectos."
        }
    ]

if "historial_busqueda" not in st.session_state:
    st.session_state.historial_busqueda = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==============================================================================
# 2. MOTOR COMPARTIDO DE VECTORIZACIÓN Y CLASIFICACIÓN (TF-IDF + Naive Bayes)
# ==============================================================================
df = pd.DataFrame(st.session_state.docs)
stop_words_es = ["de", "y", "en", "para", "con", "la", "del", "el", "los", "las", "un", "una", "por", "sobre", "al"]

vectorizer = TfidfVectorizer(stop_words=stop_words_es, ngram_range=(1, 2))
texto_completo = df["titulo"] + " " + df["contenido"] + " " + df["conceptos"].apply(lambda x: " ".join(x))
X_vectores = vectorizer.fit_transform(texto_completo)

clasificador = MultinomialNB(alpha=0.5)
clasificador.fit(X_vectores, df["categoria"])

# ==============================================================================
# INTERFAZ DE USUARIO INTEGRADA
# ==============================================================================
st.title("☁️ CloudSync Engineering — Sistema de Gestión del Conocimiento (KMS)")
st.markdown("Plataforma integrada de **Clasificación Automatizada**, **Recomendación Semántica**, **Chatbot Documental** y **Grafo de Conocimiento**.")

with st.sidebar:
    st.header("📊 Estado del Repositorio")
    st.metric("Documentos Indexados", len(df))
    st.metric("Dominios / Categorías", df["categoria"].nunique())
    st.write("**Distribución por Área:**")
    st.dataframe(df["categoria"].value_counts().reset_index().rename(columns={"categoria": "Dominio", "count": "Docs"}), hide_index=True)
    if st.session_state.historial_busqueda:
        st.divider()
        st.write("**🕒 Historial de Sesión:**")
        for h in st.session_state.historial_busqueda[-5:]:
            st.caption(f"• {h}")

tabs = st.tabs([
    "1️⃣ Clasificación e Ingesta",
    "2️⃣ Sistema de Recomendación",
    "3️⃣ Chatbot Documental",
    "4️⃣ Grafo de Conocimiento",
    "5️⃣ Evaluación y Métricas"
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: CLASIFICACIÓN AUTOMATIZADA DE DOCUMENTOS
# ------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Clasificación Automatizada e Integración al Repositorio")
    col1, col2 = st.columns([1, 1.3])
    with col1:
        with st.form("form_nuevo_doc"):
            nuevo_titulo = st.text_input("Título del documento", placeholder="Ej. Guía de despliegue de contenedores en Azure AKS")
            nuevo_tipo = st.selectbox("Tipo de Documento (Ontología)", ["Manual de Configuración", "Repositorio de Código", "Reporte de Incidente", "Guía de Inducción"])
            nuevo_contenido = st.text_area("Resumen o contenido técnico", placeholder="Describe el procedimiento, arquitectura o política...")
            nuevos_conceptos = st.text_input("Conceptos clave (separados por coma)", "Microsoft Azure, Kubernetes, Despliegue")
            submitted = st.form_submit_button("Clasificar y Agregar al Sistema")

            if submitted and nuevo_titulo and nuevo_contenido:
                vec_nuevo = vectorizer.transform([nuevo_titulo + " " + nuevo_contenido + " " + nuevos_conceptos])
                cat_predicha = clasificador.predict(vec_nuevo)[0]
                probs = clasificador.predict_proba(vec_nuevo)[0]
                confianza = np.max(probs) * 100

                nuevo_id = f"CS-{len(st.session_state.docs)+1:02d}"
                st.session_state.docs.append({
                    "id": nuevo_id,
                    "titulo": nuevo_titulo,
                    "tipo": nuevo_tipo,
                    "categoria": cat_predicha,
                    "conceptos": [c.strip() for c in nuevos_conceptos.split(",") if c.strip()],
                    "contenido": nuevo_contenido
                })
                st.success(f"✅ Documento **{nuevo_id}** clasificado automáticamente en **{cat_predicha}** (Confianza: {confianza:.1f}%). Ya está disponible en el Recomendador, Chatbot y Grafo.")
                st.rerun()

    with col2:
        st.write("**Repositorio Central Activo:**")
        st.dataframe(df[["id", "titulo", "tipo", "categoria", "conceptos"]], use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# PESTAÑA 2: SISTEMA DE RECOMENDACIÓN (Similitud del Coseno + Historial)
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Recomendación de Documentos Relacionados")
    modo_rec = st.radio("Criterio de recomendación:", ["Por documento seleccionado", "Por historial de consultas de la sesión"], horizontal=True)

    if modo_rec == "Por documento seleccionado":
        doc_seleccionado = st.selectbox("Selecciona un documento base:", df["titulo"])
        idx_base = df[df["titulo"] == doc_seleccionado].index[0]

        if doc_seleccionado not in st.session_state.historial_busqueda:
            st.session_state.historial_busqueda.append(doc_seleccionado)

        similitudes = cosine_similarity(X_vectores[idx_base], X_vectores).flatten()
        df_rec = df.copy()
        df_rec["similitud"] = similitudes
        recomendados = df_rec[df_rec.index != idx_base].sort_values(by="similitud", ascending=False).head(3)

        st.markdown(f"**Documento base:** `{df.loc[idx_base, 'id']}` — **{doc_seleccionado}** *(Dominio: {df.loc[idx_base, 'categoria']})*")
        st.write("**Documentos recomendados (Top 3 por Similitud del Coseno):**")

        for _, row in recomendados.iterrows():
            conceptos_comunes = set(df.loc[idx_base, "conceptos"]).intersection(set(row["conceptos"]))
            criterio_txt = f"Conceptos compartidos: {', '.join(conceptos_comunes)}" if conceptos_comunes else f"Proximidad léxica en dominio {row['categoria']}"
            with st.container(border=True):
                st.markdown(f"📄 **{row['titulo']}** (`{row['id']}`) — **Similitud: {row['similitud']*100:.2f}%**")
                st.caption(f"Categoría: {row['categoria']} | Tipo: {row['tipo']} | Evidencia: {criterio_txt}")
                st.write(row["contenido"])
    else:
        if not st.session_state.historial_busqueda:
            st.info("Aún no tienes consultas en el historial. Selecciona un documento arriba o realiza una pregunta en el Chatbot.")
        else:
            contexto_historial = " ".join(st.session_state.historial_busqueda)
            st.write(f"**Contexto acumulado de sesión:** `{contexto_historial}`")
            vec_hist = vectorizer.transform([contexto_historial])
            sims_hist = cosine_similarity(vec_hist, X_vectores).flatten()
            df_hist = df.copy()
            df_hist["similitud"] = sims_hist
            for _, row in df_hist.sort_values(by="similitud", ascending=False).head(3).iterrows():
                with st.container(border=True):
                    st.markdown(f"📄 **{row['titulo']}** (`{row['id']}`) — **Relevancia contextual: {row['similitud']*100:.2f}%**")
                    st.caption(f"Categoría: {row['categoria']} | Conceptos: {', '.join(row['conceptos'])}")

# ------------------------------------------------------------------------------
# PESTAÑA 3: CHATBOT PARA CONSULTAS EN LENGUAJE NATURAL
# ------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Asistente Virtual de Conocimiento (CloudSync Bot)")
    st.caption("Consulta procedimientos técnicos, incidentes, APIs o políticas internas en lenguaje natural.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pregunta = st.chat_input("Ej. ¿Qué hacemos si cae la base de datos SQL o cómo configuro AWS?")
    if pregunta:
        st.session_state.historial_busqueda.append(pregunta)
        st.session_state.chat_history.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        vec_preg = vectorizer.transform([pregunta])
        scores = cosine_similarity(vec_preg, X_vectores).flatten()
        top_indices = scores.argsort()[::-1][:2]

        if scores[top_indices[0]] > 0.05:
            doc1 = df.iloc[top_indices[0]]
            score1 = scores[top_indices[0]] * 100
            respuesta = (
                f"Con base en el repositorio de **CloudSync Engineering**, encontré la siguiente información relevante:\n\n"
                f"1. **Fuente principal:** *{doc1['titulo']}* (`{doc1['id']}` — Dominio: **{doc1['categoria']}**, Similitud: **{score1:.1f}%**)\n"
                f"   > {doc1['contenido']}\n\n"
            )
            if scores[top_indices[1]] > 0.05:
                doc2 = df.iloc[top_indices[1]]
                score2 = scores[top_indices[1]] * 100
                respuesta += (
                    f"2. **Documento complementario:** *{doc2['titulo']}* (`{doc2['id']}` — Dominio: **{doc2['categoria']}**, Similitud: **{score2:.1f}%**)\n"
                    f"   > {doc2['contenido']}"
                )
        else:
            respuesta = "No encontré documentos con suficiente coincidencia semántica en el repositorio de CloudSync Engineering. Intenta reformular usando términos como *AWS, Azure, SQL, Docker, API REST, onboarding o vacaciones*."

        st.session_state.chat_history.append({"role": "assistant", "content": respuesta})
        with st.chat_message("assistant"):
            st.markdown(respuesta)

# ------------------------------------------------------------------------------
# PESTAÑA 4: GRAFO DE CONOCIMIENTO INTERACTIVO (Expansión de Ontología Fase I)
# ------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Visualización Interactiva de la Ontología de CloudSync Engineering")
    st.markdown("🔵 **Documentos** | 🔴 **Dominios / Categorías** | 🟢 **Conceptos Clave** | 🟠 **Tipos de Documento**")

    filtro_dom = st.multiselect("Filtrar por Dominio:", df["categoria"].unique(), default=list(df["categoria"].unique()))
    df_grafo = df[df["categoria"].isin(filtro_dom)]

    G = nx.DiGraph()

    # Relaciones estructurales de la Ontología Fase I
    relaciones_ontologia = [
        ("Desarrollo", "Inteligencia Artificial", "se_integra_con"),
        ("Infraestructura", "Desarrollo", "soporta_despliegue_de")
    ]
    for src, dst, rel in relaciones_ontologia:
        if src in filtro_dom and dst in filtro_dom:
            G.add_edge(src, dst, label=rel, color="#888888")

    for _, row in df_grafo.iterrows():
        # Nodo Dominio (Categoría)
        G.add_node(row["categoria"], color="#E94E77", size=30, title=f"Dominio: {row['categoria']}")
        # Nodo Documento
        G.add_node(row["id"], label=f"{row['id']}: {row['titulo'][:25]}...", color="#4A90E2", size=20, title=f"{row['titulo']} ({row['tipo']})")
        # Nodo Tipo de Documento
        G.add_node(row["tipo"], color="#F5A623", size=22, title=f"Tipo Ontológico: {row['tipo']}")

        G.add_edge(row["id"], row["categoria"], label="pertenece_a")
        G.add_edge(row["id"], row["tipo"], label="es_un")

        for concepto in row["conceptos"]:
            G.add_node(concepto, color="#50E3C2", size=15, title=f"Concepto Técnico: {concepto}")
            G.add_edge(row["id"], concepto, label="aborda")

    net = Network(height="520px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
    net.from_nx(G)
    net.repulsion(node_distance=160, spring_length=140)
    net.save_graph("grafo_cloudsync.html")
    with open("grafo_cloudsync.html", "r", encoding="utf-8") as f:
        components.html(f.read(), height=540)

# ------------------------------------------------------------------------------
# PESTAÑA 5: EVALUACIÓN DE RENDIMIENTO (Precisión y Exhaustividad)
# ------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Evaluación Cuantitativa del Clasificador (Conjunto de Prueba)")
    st.write("Resultados obtenidos sobre un conjunto de evaluación etiquetado de 8 documentos nuevos de CloudSync Engineering:")

    test_docs = [
        ("Configuración de balanceador ALB y subredes en AWS", "Infraestructura"),
        ("Post-mortem por caída de servidor Linux en Azure", "Infraestructura"),
        ("Política de reembolso de certificaciones y vacaciones", "Recursos Humanos"),
        ("Manual de inducción y entrega de equipo para nuevos ingresos", "Recursos Humanos"),
        ("Endpoints RESTful en Python con autenticación por token", "Desarrollo"),
        ("Contenedores Docker y despliegue en clúster Kubernetes", "Desarrollo"),
        ("Entrenamiento de modelo predictivo y pipeline MLOps", "Inteligencia Artificial"),
        ("Evaluación de métricas de precisión en modelos NLP", "Inteligencia Artificial")
    ]

    X_test = vectorizer.transform([d[0] for d in test_docs])
    y_true = [d[1] for d in test_docs]
    y_pred = clasificador.predict(X_test)

    clases = sorted(list(set(y_true)))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=clases, zero_division=0)

    df_metricas = pd.DataFrame({
        "Dominio / Categoría": clases,
        "Precisión (Precision)": [f"{p*100:.1f}%" for p in prec],
        "Exhaustividad (Recall)": [f"{r*100:.1f}%" for r in rec],
        "F1-Score": [f"{f*100:.1f}%" for f in f1]
    })
    st.table(df_metricas)
