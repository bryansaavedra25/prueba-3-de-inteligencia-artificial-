# prueba-3-de-inteligencia-artificial-

 Ingeniería de Inteligencia Artificial – EP2 & EP3
Agente Inteligente RAG con Memoria, Planificación, Observabilidad y Toma de Decisiones – Banco Andino

Este proyecto corresponde a las Evaluaciones EP2 y EP3 de la asignatura Ingeniería de Inteligencia Artificial.
Se desarrolla un agente autónomo, con capacidades avanzadas de:

 RAG (Recuperación aumentada con documentos)
 Memoria (corta y larga)
 Planificación (TaskPlanner)
 Toma de decisiones adaptativa
 Observabilidad y métricas (Prometheus)
 Dashboard de análisis (Streamlit)
 Logging estructurado y trazabilidad (JSONL)

Implementado con FastAPI, LangChain, FAISS, HuggingFace Embeddings y Prometheus Client.

 A. Diseño e Implementación del Agente (IE1, IE2)

El agente se diseñó con arquitectura modular para simular un asistente interno del Banco Andino, capaz de:

Recuperar información desde documentos internos.

Razonar sobre la normativa financiera chilena (CMF).

Detectar datos sensibles y derivar cuando corresponde.

Registrar notas operacionales y trazas auditables.

Ejecutar un pipeline inteligente paso a paso.

Se implementaron tres herramientas principales:

1. search_docs

Busca información semántica usando FAISS + embeddings MiniLM.

 2. reason_policy

Decide si:

se responde,

o se deriva a un ejecutivo.

Basado en:

presencia de datos sensibles,

existencia de contexto,

reglas internas.

 3. write_note

Registra evidencias operacionales en formato JSONL, almacenadas en /data/notas_operacionales.jsonl.

Estas herramientas se combinan dentro de un pipeline inteligente que ejecuta:
→ Seguridad → Recuperación de contexto → Razonamiento → Respuesta/Derivación → Registro

B. Memoria y Recuperación de Contexto (IE3, IE4)

Se implementó un sistema de memoria en dos niveles:

1. Memoria Corta (ShortMemory)

Guarda las últimas 10 interacciones.

Permite coherencia en diálogos largos.

Se expone desde /memoria/corto.

 2. Memoria Larga (FAISS VectorStore)

Embeddings generados con HuggingFace.

Búsqueda semántica profunda.

Se carga automáticamente al iniciar la API.

Gracias a esta arquitectura híbrida, el agente:
✔ fundamenta respuestas,
✔ mantiene coherencia,
✔ recuerda contexto previo.

 C. Planificación y Toma de Decisiones Adaptativa (IE5, IE6)

La clase TaskPlanner define el orden de ejecución:

["seguridad", "recuperar_ctx", "razonar", "responder", "registrar"]


El agente adapta su comportamiento según:

si la consulta contiene datos sensibles,

si existe información relevante en FAISS,

si el LLM genera errores,

si debe derivar o responder.

El pipeline completo queda registrado en:
/data/traces_ep2.log
(extendido para EP3).

D. Observabilidad y Métricas – EP3 (IE1, IE2)

La Evaluación 3 exigía agregar instrumentación completa, incluyendo:

✔ Métricas Prometheus implementadas
Tipo	Métrica	Descripción
Counter	rag_requests_total	Total de consultas procesadas (canal/decisión/sensible)
Histogram	rag_request_latency_seconds	Tiempo total del endpoint /consultar
Histogram	rag_llm_latency_seconds	Latencia del llamado al RAG/LLM
Gauge	system_cpu_percent	Uso actual de CPU del agente
Gauge	system_memory_percent	Uso actual de RAM

Endpoint expuesto:

GET /metrics


Prometheus puede leer estas métricas directamente.

 E. Trazabilidad y Logging EP3 (IE7, IE8)

Cada consulta genera una traza completa:

request_id

pasos del planner

latencias por herramienta

decisión (responder/derivar/error)

short_memory

documentos fuente

errores

canal

Estas trazas se guardan en:

/data/traces_ep3.jsonl


(diferente al EP2 para uso del dashboard)

 F. Dashboard EP3 (Streamlit)

Se creó el dashboard:

dashboards/streamlit_dashboard.py


Que muestra:

Total de consultas

Latencia promedio

Tokens consumidos

Top decisiones

Top canales

Latencia por etapa (search_docs, LLM, total)

Trazas visualizadas

Uso de CPU y RAM

 Capturas del Dashboard EP3

(Agregadas por el estudiante según instrucción del profesor)




G. Endpoints del Sistema (IE9)
Método	Ruta	Descripción
GET	/salud	Estado del sistema
POST	/consultar	Ejecuta el agente completo
GET	/memoria/corto	Últimas 10 interacciones
POST	/nota	Registra una nota manual
GET	/metrics	Métricas Prometheus
 H. Documentación Técnica (IE7, IE8)

Se incluyen:

1. Diagrama de Arquitectura del Agente

Flujo de tareas

Interacción entre módulos

VectorStore + LLM + Planner

 2. Diagrama del Pipeline RAG

Orquestación

Herramientas

Memoria + decisiones

Ambos se encuentran en /docs.

I. Referencias Técnicas (APA)

FastAPI. (2025). FastAPI Framework Documentation. https://fastapi.tiangolo.com

LangChain. (2025). LangChain Framework – Agents & RAG Documentation. https://python.langchain.com

HuggingFace. (2025). Sentence Transformers: all-MiniLM-L6-v2. https://huggingface.co/sentence-transformers

FAISS. (2025). Facebook AI Similarity Search. https://faiss.ai

OpenAI. (2025). Chat Models API Reference. https://platform.openai.com/docs

Prometheus. (2025). Prometheus Client Python Documentation. https://prometheus.io/docs

Streamlit. (2025). Streamlit Documentation. https://streamlit.io
