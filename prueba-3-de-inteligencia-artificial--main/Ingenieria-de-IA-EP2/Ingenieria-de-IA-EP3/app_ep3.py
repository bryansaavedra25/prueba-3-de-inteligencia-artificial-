from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal, Tuple
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import os
import logging
import uuid
import json
from datetime import datetime
# +++ NUEVAS IMPORTACIONES PARA OBSERVABILIDAD Y METRICAS +++
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge
from starlette_exporter import PrometheusMiddleware, handle_metrics

# -------------------- LangChain / LLM / RAG --------------------
# Nota: evita versiones obsoletas. Requiere:
#   pip install fastapi uvicorn langchain langchain-community langchain-openai faiss-cpu pydantic==2.* sentence-transformers python-dotenv prometheus-client starlette-exporter psutil
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -------------------- CONFIGURACIÓN --------------------
CARPETA_DOCUMENTOS = os.getenv("DOCS_DIR", "./docs")               # directorio con .txt/.md
RUTA_VECTORSTORE = os.getenv("VECTOR_DIR", "./vectorstore/faiss_index")
MODELO_EMBEDDING = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MODELO_LLM = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
K_VECINOS = int(os.getenv("TOP_K", "4"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
# Seguridad: NO hardcodear secretos. Usa variables de entorno:
#   OPENAI_API_KEY=...
# ------------------------------------------------------------------------------------

PALABRAS_SENSIBLES = [
    "clave", "password", "contraseña", "número de tarjeta", "cvv", "rut", "domicilio",
    "tarjeta de crédito", "nro de cuenta", "código de verificación"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("BancoAndinoRAG_EP2")

# +++ NUEVA CONFIGURACIÓN PARA LOGGING Y RUTA UNIFICADA (EP3) +++
NOTES_FILE = os.getenv("NOTES_FILE", "./data/notas_operacionales.jsonl")
LOG_FILE = os.getenv("LOG_FILE", "./logs/ep3_logs.jsonl") # Ruta unificada para logs estructurados

# -------------------- MODELOS API --------------------
# ... (modelos existentes) ...

# -------------------- UTILIDADES DE SEGURIDAD --------------------
# ... (funciones existentes) ...

# -------------------- DOCUMENTOS & VECTORSTORE --------------------
# ... (funciones existentes) ...

# -------------------- PROMPTS --------------------
# ... (prompts y función existente plantilla_para) ...

# -------------------- HERRAMIENTAS (IE1) --------------------
# Tool 1: search_docs
def tool_search_docs(query: str, retriever, k: int = K_VECINOS) -> Tuple[str, List[Any]]:
    # ... (cuerpo de la función existente) ...
    results = retriever.get_relevant_documents(query)
    top_docs = results[:k]
    context = "\n\n".join([d.page_content for d in top_docs])
    return context, top_docs

# Tool 2: write_note
os.makedirs(os.path.dirname(NOTES_FILE), exist_ok=True) if "/" in NOTES_FILE else None

def tool_write_note(titulo: str, contenido: str) -> Dict[str, Any]:
    note = {
        "id": str(uuid.uuid4()),
        "titulo": titulo,
        "contenido": contenido,
        "ts": datetime.utcnow().isoformat()
    }
    with open(NOTES_FILE, "a", encoding="utf8") as f:
        f.write(json.dumps(note, ensure_ascii=False) + "\n")
    return note

# Tool 3: reason_policy
# ... (cuerpo de la función existente) ...

# -------------------- MEMORIA --------------------
# ... (clase ShortMemory y objeto SHORT_MEMORY existente) ...
VECTORSTORE_GLOBAL: Optional[FAISS] = None

# -------------------- PLANNER --------------------
# ... (clase TaskPlanner y objeto PLANNER existente) ...

# -------------------- CONSTRUCCIÓN RAG --------------------
# ... (función construir_chain_rag y objeto CHAIN_RAG_GLOBAL existente) ...

# -------------------- PROMETHEUS METRICS (IE1, IE2) --------------------
# Se inicializan las métricas para la trazabilidad y observabilidad
rag_requests_total = Counter(
    "rag_requests_total",
    "Total de consultas procesadas por el agente RAG",
    ["decision", "canal", "sensible"]
)

rag_request_latency_seconds = Histogram(
    "rag_request_latency_seconds",
    "Latencia total del endpoint /consultar (segundos)",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
)

rag_llm_latency_seconds = Histogram(
    "rag_llm_latency_seconds",
    "Latencia del llamado al RAG/LLM (segundos)",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Métricas del sistema (Gauge) para uso de recursos
system_cpu_percent = Gauge("system_cpu_percent", "Uso actual de CPU del proceso del agente")
system_memory_percent = Gauge("system_memory_percent", "Uso actual de RAM del proceso del agente")
PROCESS = psutil.Process(os.getpid())

# -------------------- FASTAPI --------------------
app = FastAPI(title="Banco Andino - Agente RAG (EP2/EP3)")

# Se agrega el middleware de Prometheus para exponer el endpoint /metrics
app.add_middleware(PrometheusMiddleware, app_name="banco_andino_rag")
app.add_route("/metrics", handle_metrics)

@app.on_event("startup")
async def init():
    global VECTORSTORE_GLOBAL, CHAIN_RAG_GLOBAL
    try:
        if os.path.exists(RUTA_VECTORSTORE):
            VECTORSTORE_GLOBAL = cargar_vectorstore_guardado(RUTA_VECTORSTORE)
        else:
            VECTORSTORE_GLOBAL = construir_y_guardar_vectorstore(CARPETA_DOCUMENTOS, RUTA_VECTORSTORE)
        CHAIN_RAG_GLOBAL = construir_chain_rag(VECTORSTORE_GLOBAL)
        # +++ Asegurar la creación del directorio de logs +++
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True) if "/" in LOG_FILE else None
        logger.info("EP3: Directorio de logs creado en: %s", os.path.dirname(LOG_FILE))
        logger.info("EP2: Sistema inicializado.")
    except Exception as e:
        logger.exception("EP2: Error inicializando: %s", e)


@app.get("/salud")
async def salud():
    listo = VECTORSTORE_GLOBAL is not None and CHAIN_RAG_GLOBAL is not None
    return {"status": "ok" if listo else "inicializando"}


# -------------------- PIPELINE EP2 (IE6: decisiones adaptativas) --------------------
@app.post("/consultar", response_model=RespuestaConsulta)
async def consultar(solicitud: SolicitudConsulta):
    global CHAIN_RAG_GLOBAL, VECTORSTORE_GLOBAL
    start_time = time.time() # Iniciar contador de latencia total
    is_sensible = "false"
    tokens_consumed = 0

    if CHAIN_RAG_GLOBAL is None or VECTORSTORE_GLOBAL is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible - inicialización en curso")

    req_id = str(uuid.uuid4())
    pregunta = solicitud.pregunta.strip()
    tooltrace: List[Dict[str, Any]] = []
    planner_trace: Dict[str, Any] = {"plan": PLANNER.plan(pregunta), "decisiones": []}

    # Paso 1: Seguridad
    if contiene_informacion_sensible(pregunta):
        is_sensible = "true"
        decision = "derivar"
        planner_trace["decisiones"].append({"paso": "seguridad", "accion": "derivar_por_datos_sensibles"})
        respuesta = ("⚠️ La consulta contiene datos sensibles. Por seguridad será derivada a un ejecutivo.")
        
        SHORT_MEMORY.add("user", pregunta)
        SHORT_MEMORY.add("assistant", respuesta)

        # +++ Registro de métricas y traza final para casos sensibles +++
        end_time = time.time()
        latency_total = end_time - start_time
        
        rag_requests_total.labels(decision=decision, canal=solicitud.canal or "web", sensible=is_sensible).inc()
        rag_request_latency_seconds.observe(latency_total)

        traza = {
            "ts": datetime.utcnow().isoformat(),
            "request_id": req_id,
            "cliente_id": solicitud.cliente_id,
            "canal": solicitud.canal,
            "pregunta": pregunta,
            "decision": decision,
            "latencia_total": latency_total,
            "planner": planner_trace,
            "tooltrace": tooltrace,
            "short_memory_tail": SHORT_MEMORY.as_text()[-500:]
        }
        with open(LOG_FILE, "a", encoding="utf8") as f:
            f.write(json.dumps(traza, ensure_ascii=False) + "\n") # Usar LOG_FILE
        
        return RespuestaConsulta(
            request_id=req_id, respuesta=respuesta, documentos_fuente=[],
            planner=planner_trace, tooltrace=tooltrace
        )
    
    # +++ Asignar decisión inicial para el path "No sensible" +++
    decision = "responder" 

    # Paso 2: Recuperar contexto (tool_search_docs)
    retriever = VECTORSTORE_GLOBAL.as_retriever(search_type="similarity", search_kwargs={"k": K_VECINOS})
    contexto, docs = tool_search_docs(pregunta, retriever, k=K_VECINOS)
    ctx_found = len(docs) > 0 and len(contexto.strip()) > 0
    tooltrace.append({"tool": "search_docs", "k": K_VECINOS, "ctx_found": ctx_found})

    # Paso 3: Razonar política (derivar vs responder)
    decision = tool_reason_policy(pregunta, ctx_found)
    planner_trace["decisiones"].append({"paso": "razonar", "decision": decision, "ctx_found": ctx_found})

    # Paso 4: Generar respuesta (si aplica)
    documentos_fuente: List[DocumentoFuente] = []
    respuesta_texto = ""
    llm_start_time = None

    if decision == "responder":
        plantilla = plantilla_para(pregunta)
        prompt = plantilla.format(question=pregunta)
        try:
            llm_start_time = time.time()
            # Usamos CHAIN_RAG para fundamentar en fuentes
            resultado = CHAIN_RAG_GLOBAL({"query": pregunta})
            llm_end_time = time.time()
            rag_llm_latency_seconds.observe(llm_end_time - llm_start_time) # Registrar latencia LLM

            salida = resultado.get("result") if isinstance(resultado, dict) else str(resultado)
            src_docs = resultado.get("source_documents", []) if isinstance(resultado, dict) else []

            # Sanitizar salida por si el LLM generó algo sensible
            respuesta_texto = sanitizar(salida)

            for d in src_docs:
                contenido = getattr(d, "page_content", "")[:600]
                score = float(d.metadata.get("score", 0.0)) if isinstance(d.metadata, dict) else 0.0
                documentos_fuente.append(
                    DocumentoFuente(contenido_parcial=contenido, puntuacion=score, metadatos=d.metadata)
                )

            tooltrace.append({"tool": "llm_rag", "ok": True})
            # Estimación simple de tokens consumidos
            tokens_consumed = len(pregunta.split()) + len(respuesta_texto.split()) + len(contexto.split())
        except Exception as e:
            logger.exception("Error generando respuesta RAG: %s", e)
            respuesta_texto = ("Ocurrió un error al procesar la consulta. Será derivada a un ejecutivo.")
            decision = "derivar"
            tooltrace.append({"tool": "llm_rag", "ok": False, "error": str(e)})

    if decision == "derivar":
        # respuesta de derivación estándar
        if not respuesta_texto:
            respuesta_texto = ("Gracias por tu consulta. Para resguardar tus datos y darte una respuesta precisa, "
                               "derivaremos este caso a un ejecutivo del Banco Andino.")

    # Paso 5: Registrar nota operativa (write_note) con resumen (no datos sensibles)
    try:
        resumen = (pregunta[:160] + "...") if len(pregunta) > 160 else pregunta
        note = tool_write_note("Consulta cliente", f"req={req_id} canal={solicitud.canal} resumen='{resumen}' decision={decision}")
        tooltrace.append({"tool": "write_note", "ok": True, "note_id": note["id"]})
    except Exception as e:
        tooltrace.append({"tool": "write_note", "ok": False, "error": str(e)})

    # Memoria corto plazo
    SHORT_MEMORY.add("user", pregunta)
    SHORT_MEMORY.add("assistant", respuesta_texto)

    # +++ Instrumentación final (Prometheus y JSONL) +++
    end_time = time.time()
    latency_total = end_time - start_time
    
    # 1. Prometheus
    rag_requests_total.labels(decision=decision, canal=solicitud.canal or "web", sensible=is_sensible).inc()
    rag_request_latency_seconds.observe(latency_total)
    
    # System metrics
    system_cpu_percent.set(PROCESS.cpu_percent(interval=None))
    system_memory_percent.set(PROCESS.memory_percent())

    # 2. Trazas en archivo (auditoría)
    traza = {
        "ts": datetime.utcnow().isoformat(),
        "request_id": req_id,
        "cliente_id": solicitud.cliente_id,
        "canal": solicitud.canal,
        "pregunta": pregunta,
        "decision": decision,
        "latencia_total": latency_total,
        "tokens": tokens_consumed,
        "planner": planner_trace,
        "tooltrace": tooltrace,
        "short_memory_tail": SHORT_MEMORY.as_text()[-500:]
    }
    with open(LOG_FILE, "a", encoding="utf8") as f: # Usar LOG_FILE
        f.write(json.dumps(traza, ensure_ascii=False) + "\n")

    return RespuestaConsulta(
        request_id=req_id,
        respuesta=respuesta_texto,
        documentos_fuente=documentos_fuente,
        planner=planner_trace,
        tooltrace=tooltrace
    )


# Endpoint para registrar notas manualmente (herramienta de escritura) – evidencia IE1
@app.post("/nota")
async def crear_nota(nota: SolicitudNota):
# ... (cuerpo de la función existente) ...
    try:
        created = tool_write_note(nota.titulo, nota.contenido)
        return {"ok": True, "note": created}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando nota: {e}")


# Endpoint para inspeccionar breve estado de memoria corta – evidencia IE3
@app.get("/memoria/corto")
async def memoria_corto():
# ... (cuerpo de la función existente) ...
    return {"turns": SHORT_MEMORY.buffer}

# ------------------------------------------------------------------------------------
# Dev server:
# uvicorn app_ep3:app --host 0.0.0.0 --port 8000 --reload
# ------------------------------------------------------------------------------------
