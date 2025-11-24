# prueba-3-de-inteligencia-artificial-

 Evaluación 3 – Informe Técnico
Sección: Observabilidad y Trazabilidad con LangSmith (EP3)

En esta sección se presenta la evidencia de observabilidad requerida por la Evaluación 3, utilizando la plataforma LangSmith como sistema de tracking, visualización y análisis de instrumentación del agente inteligente RAG del Banco Andino.

Gracias a la activación de:

LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=xxxx
LANGSMITH_PROJECT="BancoAndino_EP3"


el agente envía a LangSmith todas las trazas del pipeline, incluyendo:

llamadas a herramientas,

pasos del planificador (TaskPlanner),

latencias individuales,

contexto recuperado,

prompts enviados al LLM,

tokens consumidos,

errores y derivaciones.

A continuación, se presenta evidencia visual correspondiente a cada parte del proceso.

🖼️ 1. Trazabilidad Completa del Pipeline del Agente (IE1, IE2, IE6)

En esta captura se observa la ejecución completa del pipeline del endpoint /consultar, mostrando la secuencia:

seguridad

recuperar_ctx (search_docs)

razonar (reason_policy)

responder (llm_rag)

registrar (write_note)

Además, LangSmith permite ver el trace tree, donde se refleja con exactitud:

las llamadas al vectorstore FAISS,

la construcción del prompt final,

la ejecución del LLM,

los documentos fuente,

el contenido generado,

los metadatos del sistema.

📸 Colocar captura aquí:

docs/langsmith_trace_pipeline.png


Ejemplo sugerido:

🖼️ 2. Captura: Llamadas a Herramientas (search_docs, reason_policy, write_note) – IE1

Esta captura muestra específicamente las herramientas del agente utilizadas durante el pipeline:

search_docs → ejecución del retrieval FAISS

reason_policy → decisión responder/derivar

write_note → registro operativo en JSONL

En LangSmith se observan:

inputs de cada herramienta

outputs

latencia individual

jerarquía de ejecución

📸 Colocar captura aquí:

docs/langsmith_tools.png


Ejemplo:

🖼️ 3. Captura: Ejecución del Modelo LLM con RAG (IE2, IE6)

En esta sección se ve:

el prompt enviado al LLM,

los documentos recuperados por FAISS,

contenido del contexto,

tokens consumidos,

latencia exacta del LLM,

respuesta generada.

Esta evidencia demuestra la instrumentación del RAG solicitada en la evaluación.

📸 Imagen sugerida:

docs/langsmith_llm_run.png


Ejemplo:

🖼️ 4. Métricas: Tokens, Latencia, Estados y Errores (IE1, IE2)

LangSmith proporciona métricas detalladas por ejecución:

total tokens

prompt tokens

completion tokens

latencia total

latencia por herramienta

errores del sistema (si los hubiera)

Esto cumple directamente con los indicadores de evaluabilidad IE1 e IE2.

📸 Colocar captura:

docs/langsmith_metrics.png


Ejemplo:

🖼️ 5. Vista General del Proyecto en LangSmith (Runs View) – IE7, IE8

El panel principal muestra:

todas las consultas realizadas

duración

tipo de decisión (responder / derivar)

tipo de error (si existe)

agente utilizado

fecha y hora

Esta evidencia demuestra la trazabilidad completa del agente.

📸 Colocar imagen:

docs/langsmith_runs.png


Ejemplo:

🖼️ 6. Comparación de Ejecuciones del Agente (IE9)

LangSmith permite comparar múltiples ejecuciones, mostrando:

variación de latencias

diferencias en los documentos recuperados

comportamiento del LLM bajo escenarios distintos

consistencia del razonamiento

Esto es particularmente útil para validar la robustez del agente y cumple con el requisito de evidencia comparativa del punto IE9.

📸 Colocar captura:

docs/langsmith_compare.png


Ejemplo:

🖼️ 7. Insights de Desempeño y Tiempos por Etapa – IE2

LangSmith muestra visualizaciones automáticas derivadas de los runs, como:

histogramas de latencia,

tiempo promedio por herramienta,

conteo de herramientas por ejecución,

tokens promedio,

tasa de éxito/derivación.

📸 Agregar:

docs/langsmith_insights.png


Ejemplo:

📌 Conclusión de la Sección

Las capturas obtenidas desde LangSmith demuestran que el agente:

✔ Está completamente instrumentado (IE1)
✔ Registra latencias, tokens y errores (IE2)
✔ Mantiene trazabilidad completa (IE6)
✔ Visualiza el pipeline y decisiones (IE5)
✔ Expone evidencia para auditoría y evaluación (IE7, IE8, IE9)

Con esto se cumple todo lo exigido por la Evaluación 3 sobre observabilidad, trazas y monitoreo del agente.
