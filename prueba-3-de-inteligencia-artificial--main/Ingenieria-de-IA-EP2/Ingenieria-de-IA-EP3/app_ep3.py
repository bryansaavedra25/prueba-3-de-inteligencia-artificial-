
from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal, Tuple
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import os
import logging
import uuid
import json
from datetime import datetime

# -------------------- LangChain / LLM / RAG --------------------
# Nota: evita versiones obsoletas. Requiere:
#   pip install fastapi uvicorn langchain langchain-community langchain-openai faiss-cpu pydantic==2.* sentence-transformers python-dotenv
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
#   (si usas Azure/otro endpoint, configura variables específicas del provider)
# ------------------------------------------------------------------------------------
# SENSIBLE: el código original traía tokens y URLs hardcodeadas. Se eliminaron para cumplir buenas prácticas.
# ------------------------------------------------------------------------------------

PALABRAS_SENSIBLES = [
    "clave", "password", "contraseña", "número de tarjeta", "cvv", "rut", "domicilio",
    "tarjeta de crédito", "nro de cuenta", "código de verificación"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("BancoAndinoRAG_EP2")

# -------------------- MODELOS API --------------------
class SolicitudConsulta(BaseModel):
    cliente_id: Optional[str] = None
    canal: Optional[str] = "web"
    pregunta: str


class DocumentoFuente(BaseModel):
    contenido_parcial: str
    puntuacion: float
    metadatos: Dict[str, Any]


class RespuestaConsulta(BaseModel):
    request_id: str
    respuesta: str
    documentos_fuente: List[DocumentoFuente]
    planner: Dict[str, Any]  # explica etapas/decisiones
    tooltrace: List[Dict[str, Any]]  # ejecución de herramientas


class SolicitudNota(BaseModel):
    titulo: str
    contenido: str


# -------------------- UTILIDADES DE SEGURIDAD --------------------
def contiene_informacion_sensible(texto: str) -> bool:
    t = texto.lower()
    return any(k in t for k in PALABRAS_SENSIBLES)


def sanitizar(texto: str) -> str:
    if contiene_informacion_sensible(texto):
        return ("⚠️ Lo siento, no puedo procesar ni devolver información que contenga "
                "datos personales sensibles. La consulta será derivada a un ejecutivo.")
    return texto


# -------------------- DOCUMENTOS & VECTORSTORE --------------------
def cargar_documentos_locales(carpeta: str):
    documentos = []
    if not os.path.exists(carpeta):
        return documentos
    for raiz, _, archivos in os.walk(carpeta):
        for nombre in archivos:
            if nombre.lower().endswith((".txt", ".md")):
                ruta = os.path.join(raiz, nombre)
                loader = TextLoader(ruta, encoding="utf8")
                for doc in loader.load():
                    doc.metadata = {**doc.metadata, "archivo_origen": nombre, "path": ruta}
                    documentos.append(doc)
    return documentos


def construir_y_guardar_vectorstore(carpeta_documentos: str, ruta_guardado: str) -> FAISS:
    logger.info("Indexando documentos desde: %s", carpeta_documentos)
    documentos = cargar_documentos_locales(carpeta_documentos)
    if not documentos:
        raise RuntimeError(f"No hay documentos para indexar en: {carpeta_documentos}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    trozos = splitter.split_documents(documentos)

    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDING)
    vs = FAISS.from_documents(trozos, embeddings)
    os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
    vs.save_local(ruta_guardado)
    logger.info("Índice FAISS guardado en: %s", ruta_guardado)
    return vs


def cargar_vectorstore_guardado(ruta_guardado: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDING)
    # allow_dangerous_deserialization requerido por FAISS.load_local en versiones recientes
    vs = FAISS.load_local(ruta_guardado, embeddings, allow_dangerous_deserialization=True)
    return vs


# -------------------- PROMPTS --------------------
PROMPT_BASE = (
    "Eres un asistente del Banco Andino. Responde SOLO con base en normativa financiera chilena (CMF) "
    "y documentos internos. Si no hay respuesta en las fuentes, deriva al cliente a un ejecutivo. "
    "Sé breve y claro."
)

PROMPT_CREDITO = (
    "Un cliente pregunta: '{question}'. Responde usando EXCLUSIVAMENTE políticas internas de crédito y plazos "
    "del Banco Andino presentes en la base de conocimiento. No inventes. Si falta información, deriva."
)

PROMPT_APERTURA = (
    "Un cliente consulta: '{question}'. Usa la guía oficial de apertura de productos del Banco Andino y la normativa CMF. "
    "Enumera requisitos en viñetas. Si falta información, deriva."
)

def plantilla_para(pregunta: str) -> PromptTemplate:
    q = pregunta.lower()
    if any(k in q for k in ["crédito", "vencimiento", "cuota", "pago"]):
        return PromptTemplate.from_template(PROMPT_CREDITO)
    if any(k in q for k in ["abrir cuenta", "cuenta corriente", "requisitos para abrir", "abrir una cuenta"]):
        return PromptTemplate.from_template(PROMPT_APERTURA)
    return PromptTemplate.from_template(PROMPT_BASE)


# -------------------- HERRAMIENTAS (IE1) --------------------
# Tool 1: search_docs → consulta semántica en FAISS
def tool_search_docs(query: str, retriever, k: int = K_VECINOS) -> Tuple[str, List[Any]]:
    # Usamos retriever para traer pasajes, devolvemos texto concatenado y docs crudos para trazas
    results = retriever.get_relevant_documents(query)
    top_docs = results[:k]
    context = "\\n\\n".join([d.page_content for d in top_docs])
    return context, top_docs

# Tool 2: write_note → registra notas operacionales (simulado en json)
NOTES_FILE = os.getenv("NOTES_FILE", "./data/notas_operacionales.jsonl")
os.makedirs(os.path.dirname(NOTES_FILE), exist_ok=True) if "/" in NOTES_FILE else None

def tool_write_note(titulo: str, contenido: str) -> Dict[str, Any]:
    note = {
        "id": str(uuid.uuid4()),
        "titulo": titulo,
        "contenido": contenido,
        "ts": datetime.utcnow().isoformat()
    }
    with open(NOTES_FILE, "a", encoding="utf8") as f:
        f.write(json.dumps(note, ensure_ascii=False) + "\\n")
    return note

# Tool 3: reason_policy → aplica política de decisión simple para derivar o responder
def tool_reason_policy(pregunta: str, ctx_found: bool) -> Literal["responder", "derivar"]:
    # regla básica: si contiene datos sensibles o no hay contexto -> derivar
    if contiene_informacion_sensible(pregunta) or not ctx_found:
        return "derivar"
    return "responder"


# -------------------- MEMORIA (IE3: corto plazo | IE4: largo plazo) --------------------
# Corto plazo: buffer de la conversación (en memoria)
class ShortMemory:
    def __init__(self, max_turns: int = 10):
        self.buffer: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add(self, role: Literal["user", "assistant"], text: str):
        self.buffer.append({"role": role, "text": text})
        if len(self.buffer) > self.max_turns:
            self.buffer = self.buffer[-self.max_turns:]

    def as_text(self) -> str:
        return "\\n".join([f"{x['role']}: {x['text']}" for x in self.buffer])


SHORT_MEMORY = ShortMemory(max_turns=10)

# Largo plazo: vectorstore FAISS (ya implementado arriba) como memoria semántica
VECTORSTORE_GLOBAL: Optional[FAISS] = None


# -------------------- PLANNER (IE5) --------------------
# Planificador con prioridades por intención detectada
class TaskPlanner:
    PRIORIDADES = {
        "seguridad": 0,     # si hay riesgo/sensibilidad, va primero
        "recuperar_ctx": 1, # buscar evidencia en docs
        "razonar": 2,       # decidir derivar/responder
        "responder": 3,     # generar respuesta
        "registrar": 4      # escribir nota/evidencia
    }

    def plan(self, pregunta: str) -> List[str]:
        plan = ["seguridad", "recuperar_ctx", "razonar", "responder", "registrar"]
        # si la pregunta ya sugiere apertura de cuenta / crédito, mantenemos plan estándar
        return sorted(plan, key=lambda p: self.PRIORIDADES[p])


PLANNER = TaskPlanner()


# -------------------- CONSTRUCCIÓN RAG --------------------
def construir_chain_rag(vectorstore: FAISS) -> RetrievalQA:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": K_VECINOS})
    llm = ChatOpenAI(model=MODELO_LLM, temperature=TEMPERATURE)
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    return chain


CHAIN_RAG_GLOBAL: Optional[RetrievalQA] = None


# -------------------- FASTAPI --------------------
app = FastAPI(title="Banco Andino - Agente RAG (EP2)")

@app.on_event("startup")
async def init():
    global VECTORSTORE_GLOBAL, CHAIN_RAG_GLOBAL
    try:
        if os.path.exists(RUTA_VECTORSTORE):
            VECTORSTORE_GLOBAL = cargar_vectorstore_guardado(RUTA_VECTORSTORE)
        else:
            VECTORSTORE_GLOBAL = construir_y_guardar_vectorstore(CARPETA_DOCUMENTOS, RUTA_VECTORSTORE)
        CHAIN_RAG_GLOBAL = construir_chain_rag(VECTORSTORE_GLOBAL)
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
    if CHAIN_RAG_GLOBAL is None or VECTORSTORE_GLOBAL is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible - inicialización en curso")

    req_id = str(uuid.uuid4())
    pregunta = solicitud.pregunta.strip()
    tooltrace: List[Dict[str, Any]] = []
    planner_trace: Dict[str, Any] = {"plan": PLANNER.plan(pregunta), "decisiones": []}

    # Paso 1: Seguridad
    if contiene_informacion_sensible(pregunta):
        planner_trace["decisiones"].append({"paso": "seguridad", "accion": "derivar_por_datos_sensibles"})
        respuesta = ("⚠️ La consulta contiene datos sensibles. Por seguridad será derivada a un ejecutivo.")
        SHORT_MEMORY.add("user", pregunta)
        SHORT_MEMORY.add("assistant", respuesta)
        return RespuestaConsulta(
            request_id=req_id, respuesta=respuesta, documentos_fuente=[],
            planner=planner_trace, tooltrace=tooltrace
        )

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
    if decision == "responder":
        plantilla = plantilla_para(pregunta)
        prompt = plantilla.format(question=pregunta)
        try:
            # Usamos CHAIN_RAG para fundamentar en fuentes
            resultado = CHAIN_RAG_GLOBAL({"query": pregunta})
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

    # Trazas en archivo (auditoría)
    traza = {
        "ts": datetime.utcnow().isoformat(),
        "request_id": req_id,
        "cliente_id": solicitud.cliente_id,
        "canal": solicitud.canal,
        "pregunta": pregunta,
        "decision": decision,
        "planner": planner_trace,
        "tooltrace": tooltrace,
        "short_memory_tail": SHORT_MEMORY.as_text()[-500:]
    }
    with open("traces_ep2.log", "a", encoding="utf8") as f:
        f.write(json.dumps(traza, ensure_ascii=False) + "\\n")

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
    try:
        created = tool_write_note(nota.titulo, nota.contenido)
        return {"ok": True, "note": created}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando nota: {e}")


# Endpoint para inspeccionar breve estado de memoria corta – evidencia IE3
@app.get("/memoria/corto")
async def memoria_corto():
    return {"turns": SHORT_MEMORY.buffer}

# ------------------------------------------------------------------------------------
# Dev server:
# uvicorn app_ep2:app --host 0.0.0.0 --port 8000 --reload
# ------------------------------------------------------------------------------------
