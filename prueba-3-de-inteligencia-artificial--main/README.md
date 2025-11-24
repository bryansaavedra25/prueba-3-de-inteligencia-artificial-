Ingeniería de Inteligencia Artificial – EP2
Agente Inteligente RAG con Memoria, Planificación y Toma de Decisiones – Banco Andino
A. Diseño e Implementación del Agente (IE1, IE2)

En esta segunda evaluación desarrollé un agente inteligente funcional, mejorando la base del proyecto presentado en la Prueba 1.
El objetivo fue construir un sistema capaz de consultar información, razonar sobre ella y registrar resultados, simulando el comportamiento de un asistente interno del Banco Andino.

El agente fue implementado utilizando FastAPI, LangChain, FAISS y HuggingFace Embeddings, integrados de forma modular para facilitar su mantenimiento y escalabilidad.
Su función principal es responder preguntas de clientes sobre productos financieros, basándose en documentos internos del banco y normativa de la CMF.

El sistema cuenta con tres herramientas principales que permiten su funcionamiento:

search_docs → Recupera información semántica desde el índice FAISS.

reason_policy → Evalúa si la consulta puede responderse o si debe derivarse a un ejecutivo.

write_note → Registra notas y evidencias de las interacciones en formato JSONL.

Cada herramienta actúa de manera independiente, pero se integran en un flujo común gestionado por el agente.

B. Configuración de Memoria y Recuperación de Contexto (IE3, IE4)

El sistema utiliza dos niveles de memoria para garantizar coherencia en las respuestas y continuidad en la conversación:

Memoria corta (ShortMemory): almacena las últimas 10 interacciones entre el usuario y el asistente. Esto permite mantener contexto en conversaciones extendidas.

Memoria larga (FAISS): funciona como una base vectorial donde se guardan los embeddings de los documentos del banco, permitiendo una búsqueda semántica y respuestas fundamentadas en información real.

Gracias a esta estructura, el agente no solo responde preguntas puntuales, sino que también recuerda consultas anteriores y mantiene consistencia en los diálogos.

C. Planificación y Toma de Decisiones (IE5, IE6)

Se implementó una clase llamada TaskPlanner, encargada de organizar las etapas de ejecución del agente de manera ordenada y priorizada.
El flujo de planificación definido es el siguiente:

plan = ["seguridad", "recuperar_ctx", "razonar", "responder", "registrar"]


Con este esquema, el agente primero evalúa si la pregunta contiene datos sensibles, luego busca contexto relevante en FAISS, aplica razonamiento mediante reason_policy, genera la respuesta y finalmente registra la interacción con write_note.

Este mecanismo demuestra la capacidad de toma de decisiones adaptativa, ya que el comportamiento del agente varía según la naturaleza de la consulta o la información disponible.

D. Documentación Técnica y Orquestación (IE7, IE8)

Para representar gráficamente la estructura interna del sistema, se elaboraron diagramas técnicos que muestran la comunicación entre los módulos principales.
Estos diagramas se encuentran en la carpeta /docs/ y reflejan la orquestación general del agente y el flujo de ejecución de tareas.

Diagrama de Orquestación

Flujo de Ejecución del Agente

Los diagramas permiten visualizar la interacción entre FastAPI, el planificador de tareas, la memoria, y el modelo LLM, cumpliendo con la documentación técnica exigida por la pauta.

 E. Redacción Técnica (IE10)

El código fue desarrollado y documentado con lenguaje técnico claro y preciso, utilizando nombres descriptivos para las clases y funciones.
Se mantuvo una redacción coherente en el README y en los comentarios del código, con un estilo formal y profesional.
Además, se siguieron buenas prácticas de seguridad, evitando incluir claves o tokens directamente en el repositorio.

 F. Ejemplos de Flujo y Evidencias (IE9)

El sistema cuenta con endpoints funcionales que permiten probar su comportamiento de forma directa:

Método	Ruta	Descripción
GET	/salud	Verifica el estado del sistema.
POST	/consultar	Realiza una consulta y ejecuta todo el flujo del agente.
GET	/memoria/corto	Muestra las últimas interacciones registradas.
POST	/nota	Guarda una nota o evidencia de interacción.

Durante las pruebas, el sistema genera automáticamente archivos en la carpeta /data:

notas_operacionales.jsonl → Guarda las notas creadas por el agente.

traces_ep2.log → Registra las trazas internas del flujo y las decisiones tomadas.

Estas evidencias permiten validar el comportamiento adaptativo y la trazabilidad del agente.

G. Referencias Técnicas (APA)

FastAPI. (2025). FastAPI Framework Documentation. Recuperado de https://fastapi.tiangolo.com

LangChain. (2025). LangChain Framework – Agents & RAG Documentation. Recuperado de https://python.langchain.com

HuggingFace. (2025). Sentence Transformers: all-MiniLM-L6-v2. Recuperado de https://huggingface.co/sentence-transformers

FAISS. (2025). Facebook AI Similarity Search. Recuperado de https://faiss.ai

OpenAI. (2025). Chat Models API Reference. Recuperado de https://platform.openai.com/docs
