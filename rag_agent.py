
"""
Hybrid RAG Agent (PDF-only) with robust Ollama embeddings + LlamaIndex

Key fixes:
- Token-based chunking via LlamaIndex Settings/TokenTextSplitter to avoid Ollama 400 errors
  (Ollama rejects inputs that exceed the model’s context window; chunking keeps nodes small).
- Embedding model configured with `num_ctx` and Nomic task prefixes for better RAG quality.
- Truncation of memory/retrieved context before generation to keep within LLM limits.

References:
- LlamaIndex Settings & text splitter: https://llamaindex.openml.io/python/framework/module_guides/supporting_modules/settings/
- LlamaIndex OllamaEmbedding: https://developers.llamaindex.ai/python/examples/embeddings/ollama_embedding/
- Nomic embed model (task prefixes): https://huggingface.co/nomic-ai/nomic-embed-text-v1
- Ollama context length (why chunking matters): https://docs.ollama.com/context-length
"""

import os
import csv
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

# ---- LlamaIndex core & helpers ----
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings,                     # GLOBAL configuration point (LLM, embeddings, text splitter, etc.)
)
from llama_index.core.node_parser import TokenTextSplitter  # Token-aware chunking

# ---- File readers (PDF) ----
from llama_index.readers.file import PDFReader
# If you later install PyMuPDFReader (llama-index-readers-file + pymupdf),
# you can prefer it for more robust PDF parsing on Windows:
try:
    from llama_index.readers.file import PyMuPDFReader  # Optional, may not be installed
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

# ---- Ollama integrations ----
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


# ============================================================
# CONFIG / PATHS (override with env vars if needed)
# ============================================================
# Index storage directory
PDF_INDEX_DIR = os.environ.get("PDF_INDEX_DIR", "fixed_income_index")
# Simple JSON memory store
MEMORY_FILE = os.environ.get("MEMORY_FILE", "memory_store.json")

# PDF source directory (Windows-safe path). Consider a local folder to avoid OneDrive file locks.
# You can override with: setx PDF_DIR "C:\data\pdf_source"
PDF_DIR = Path(os.environ.get("PDF_DIR", r"C:\Users\BV56PV\OneDrive - ING\pdf_source"))

# Ollama base URL (default local)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding model & LLM (override via env if needed)
# For embeddings, `nomic-embed-text` is a strong general-purpose option.
# For generation, pick a model you have locally (e.g., "llama3.1:8b", "gemma:2b").
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "gemma")

# Set REBUILD_INDEX=1 to force a fresh index each run
REBUILD_INDEX = os.environ.get("REBUILD_INDEX", "0") == "1"


# ============================================================
# GLOBAL LlamaIndex SETTINGS (critical: token-based chunking)
# ============================================================
# Why: Ollama's embedding endpoint returns 400 if input exceeds the model context window.
# Fix: Split documents into manageable token chunks before embedding.
Settings.chunk_size = int(os.environ.get("CHUNK_SIZE_TOKENS", "512"))  # Start at 800; drop to 512 if needed
Settings.chunk_overlap = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "80"))
Settings.text_splitter = TokenTextSplitter(
    chunk_size=Settings.chunk_size,
    chunk_overlap=Settings.chunk_overlap,
)

# Optional: if you also want to set a different default tokenizer/LLM globally,
# you can do so via Settings (not mandatory here).


# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("rag_agent")


# ============================================================
# MEMORY SYSTEM
# ============================================================
def load_memory():
    """Load memory store (conversation, preferences, facts)."""
    if not Path(MEMORY_FILE).exists():
        return {"conversation_history": [], "user_preferences": {}, "learned_facts": []}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory):
    """Persist memory store to JSON."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=4)


def remember_conversation(user_msg, assistant_msg):
    """Append a conversation exchange."""
    memory = load_memory()
    memory["conversation_history"].append({
        "user": user_msg,
        "assistant": assistant_msg
    })
    save_memory(memory)


def remember_preference(key, value):
    """Save a user preference key=value."""
    memory = load_memory()
    memory["user_preferences"][key] = value
    save_memory(memory)


def remember_fact(fact):
    """Remember a free-form fact."""
    memory = load_memory()
    memory["learned_facts"].append(fact)
    save_memory(memory)


# ============================================================
# CSV INGESTION (optional helper)
# ============================================================
def load_csv_files(directory: Path):
    """
    Convert CSV files to plain-text Documents.
    Note: For large CSVs, consider schema-aware parsing & column filtering.
    """
    docs = []
    if not directory or not directory.exists():
        return docs

    for path in directory.glob("*.csv"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                rows = list(reader)
                text = "\n".join([", ".join(row) for row in rows])
                docs.append(Document(text=text, metadata={"source": str(path)}))
        except Exception as e:
            log.warning(f"Failed to parse CSV {path.name}: {e}")
    return docs


# ============================================================
# PDF INGESTION
# ============================================================
def load_pdfs_with_reader(pdf_dir: Path):
    """
    Load PDFs as Documents.

    - Uses `PDFReader` by default (no heavy native deps).
    - If `PyMuPDFReader` is available, prefer it for better robustness
      (tables, complex layouts). You can install it via:
        pip install llama-index-readers-file pymupdf

    Note: Large PDFs will be chunked later by the token-based splitter.
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF_DIR not found: {pdf_dir}")

    file_extractor = {".pdf": PDFReader()}
    if HAS_PYMUPDF:
        file_extractor = {".pdf": PyMuPDFReader()}  # Prefer PyMuPDF if available

    reader = SimpleDirectoryReader(
        input_dir=str(pdf_dir),
        recursive=True,
        file_extractor=file_extractor,
    )

    try:
        pdf_docs = reader.load_data()
    except Exception as e:
        log.error(f"PDF load failed: {e}")
        pdf_docs = []

    return pdf_docs


def _make_embed_model():
    """
    Configure the Ollama embedding model.

    - `num_ctx`: the *allocated* context size Ollama uses for this model.
      (The model's hard max still applies; chunking ensures we remain under it.)
    - `text_instruction` / `query_instruction`: task prefixes recommended by
      Nomic for best retrieval quality (`search_document:` / `search_query:`).

    References:
    - LlamaIndex OllamaEmbedding: https://developers.llamaindex.ai/python/examples/embeddings/ollama_embedding/
    - Nomic model card: https://huggingface.co/nomic-ai/nomic-embed-text-v1
    - Ollama context length: https://docs.ollama.com/context-length
    """
    return OllamaEmbedding(
        model_name=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
        ollama_additional_kwargs={"num_ctx": int(os.environ.get("EMBED_NUM_CTX", "2048"))},
        text_instruction="search_document:",
        query_instruction="search_query:",
    )


def build_pdf_index():
    """
    Build and persist the vector index from PDF documents.

    Critical pieces:
    - Documents are loaded (PDFReader / PyMuPDF).
    - Token-based chunking (Settings.text_splitter) is applied implicitly.
    - Embeddings are generated via Ollama with safe `num_ctx` and prefixes.
    """
    log.info("Loading PDF documents...")
    pdf_docs = load_pdfs_with_reader(PDF_DIR)

    # If you want CSVs too, set CSV_DIR env var and uncomment:
    # csv_dir = Path(os.environ.get("CSV_DIR", "")) if os.environ.get("CSV_DIR") else None
    # csv_docs = load_csv_files(csv_dir) if csv_dir else []
    # all_docs = pdf_docs + csv_docs

    all_docs = pdf_docs
    log.info(f"Total documents loaded: {len(all_docs)}")

    if not all_docs:
        # We still build an empty index for consistency, but warn the user.
        log.warning("No documents loaded. Index will be created but will have no content.")

    embed_model = _make_embed_model()

    # Apply the global token-based splitter via `transformations` to be explicit.
    index = VectorStoreIndex.from_documents(
        all_docs,
        embed_model=embed_model,
        transformations=[Settings.text_splitter],
    )

    index.storage_context.persist(persist_dir=PDF_INDEX_DIR)
    log.info(f"PDF index built and saved to '{PDF_INDEX_DIR}'.")


def load_pdf_index():
    """
    Load a previously persisted index. The same embed model config must be
    provided to ensure embeddings/queries are consistent.
    """
    storage_context = StorageContext.from_defaults(persist_dir=PDF_INDEX_DIR)
    embed_model = _make_embed_model()
    return load_index_from_storage(storage_context, embed_model=embed_model)


# ============================================================
# MEMORY-AWARE RAG QUERY (PDF ONLY)
# ============================================================
def _truncate(s: str, max_chars: int = 8000) -> str:
    """
    Light guard to avoid overlong prompts for the generation model.
    For strict control you can switch to token-based counting later.
    """
    if not s:
        return s
    return s[-max_chars:] if len(s) > max_chars else s


def ask_rag(question: str):
    """
    Query the PDF index with memory context and an Ollama LLM.

    Steps:
    - Load index + prepare LLM.
    - Build memory context (last 10 exchanges + prefs + facts).
    - Retrieve context from PDFs.
    - Compose final prompt and generate an answer.
    """
    pdf_index = load_pdf_index()

    # Ensure the Ollama LLM exists locally (pull it if needed)
    llm = Ollama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=300.0)
    memory = load_memory()

    # Memory context (last 10 exchanges + prefs + learned facts)
    history_text = "\n".join(
        [f"User: {h['user']}\nAssistant: {h['assistant']}"
         for h in memory.get("conversation_history", [])[-10:]]
    )
    prefs_text = "\n".join([f"{k}: {v}" for k, v in memory.get("user_preferences", {}).items()])
    facts_text = "\n".join(memory.get("learned_facts", []))

    memory_context = f"""
### Conversation Memory ###
{history_text}

### User Preferences ###
{prefs_text}

### Learned Facts ###
{facts_text}
""".strip()

    # Guard against huge memory blocks
    memory_context = _truncate(memory_context, max_chars=int(os.environ.get("MEMORY_CTX_MAX_CHARS", "8000")))

    # Retrieve context from PDFs (vector search)
    pdf_engine = pdf_index.as_query_engine(
        llm=llm,
        similarity_top_k=int(os.environ.get("SIMILARITY_TOP_K", "4")),
        response_mode="compact",
    )
    retrieval_resp = pdf_engine.query(question)

    # LlamaIndex Response object -> string
    pdf_context = getattr(retrieval_resp, "response", None)
    if not pdf_context:
        pdf_context = str(retrieval_resp)

    # Guard against huge retrieved context
    pdf_context = _truncate(pdf_context, max_chars=int(os.environ.get("PDF_CTX_MAX_CHARS", "8000")))

    # Final prompt that includes memory + retrieved context
    final_prompt = f"""
{memory_context}

### Relevant PDF Context ###
{pdf_context}

### User Question ###
{question}

Use all relevant context above to answer accurately.
""".strip()

    # LLMs are NOT callable; use complete() and extract .text
    completion = llm.complete(final_prompt)
    final_answer = completion.text if hasattr(completion, "text") else str(completion)

    remember_conversation(question, final_answer)
    return final_answer


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    log.info("Starting AI Support Agent (PDF-only RAG).")

    # Build index if missing or when forced
    if REBUILD_INDEX or not Path(PDF_INDEX_DIR).exists():
        log.info("Building PDF index...")
        try:
            build_pdf_index()
        except Exception as e:
            log.error(f"Failed to build index: {e}")

    print("\nPDF‑only RAG system ready. Ask questions (type 'exit' to quit).")

    while True:
        try:
            q = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if q.lower() in {"exit", "quit"}:
            break

        # Memory commands
        low = q.lower()
        if "remember that" in low:
            fact = q.split("remember that", 1)[1].strip()
            remember_fact(fact)
            print("Got it. I will remember that.")
            continue

        if "my preference is" in low and "=" in q:
            pref = q.split("my preference is", 1)[1].strip()
            key, value = pref.split("=", 1)
            remember_preference(key.strip(), value.strip())
            print("Preference saved.")
            continue

        # Normal RAG query
        try:
            answer = ask_rag(q)
            print("\nAnswer:", answer)
        except Exception as e:
            log.error(f"RAG query failed: {e}")
            print("Sorry, I hit an error. Check logs for details.")
