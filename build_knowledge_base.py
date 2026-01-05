import os
import csv
import json
import win32com.client
from datetime import datetime, timedelta

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext, load_index_from_storage


# ============================================================
# PATHS
# ============================================================
#EMAIL_INDEX_DIR = "email_index"
PDF_INDEX_DIR = "fixed_income_index"
MEMORY_FILE = "memory_store.json"

PDF_DIR = "C:/Users/anilo/OneDrive/pdf_source"
CSV_DIR = "C:/Users/anilo/OneDrive/csv_source"


# ============================================================
# MEMORY SYSTEM
# ============================================================
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {"conversation_history": [], "user_preferences": {}, "learned_facts": []}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=4)


def remember_conversation(user_msg, assistant_msg):
    memory = load_memory()
    memory["conversation_history"].append({
        "user": user_msg,
        "assistant": assistant_msg
    })
    save_memory(memory)


def remember_preference(key, value):
    memory = load_memory()
    memory["user_preferences"][key] = value
    save_memory(memory)


def remember_fact(fact):
    memory = load_memory()
    memory["learned_facts"].append(fact)
    save_memory(memory)


# ============================================================
# OUTLOOK EMAIL INGESTION
# ============================================================
def read_outlook_emails(limit=200, days=30):
    print("Reading Outlook inbox...")

    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    inbox = outlook.GetDefaultFolder(6)  # 6 = Inbox

    messages = inbox.Items
    messages.Sort("[ReceivedTime]", True)

    cutoff = datetime.now() - timedelta(days=days)
    emails = []
    count = 0

    for msg in messages:
        try:
            if msg.ReceivedTime < cutoff:
                break

            emails.append({
                "subject": msg.Subject,
                "sender": str(msg.SenderName),
                "sender_email": str(msg.SenderEmailAddress),
                "received": str(msg.ReceivedTime),
                "body": msg.Body,
            })

            count += 1
            if count >= limit:
                break

        except Exception:
            continue

    print(f"Loaded {len(emails)} emails.")
    return emails


def emails_to_documents(email_list):
    docs = []
    for e in email_list:
        text = f"""
        Subject: {e['subject']}
        From: {e['sender']} <{e['sender_email']}>
        Received: {e['received']}

        {e['body']}
        """
        docs.append(Document(text=text, metadata={"source": "outlook"}))
    return docs


def build_email_index():
    emails = read_outlook_emails()
    docs = emails_to_documents(emails)

    embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    index = VectorStoreIndex.from_documents(
        docs,
        embed_model=embed_model
    )

    index.storage_context.persist(persist_dir=EMAIL_INDEX_DIR)
    print("Email index built and saved.")


def load_email_index():
    storage_context = StorageContext.from_defaults(persist_dir=EMAIL_INDEX_DIR)
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    return load_index_from_storage(storage_context, embed_model=embed_model)


# ============================================================
# PDF INGESTION
# ============================================================
def load_csv_files(directory):
    docs = []
    if not os.path.exists(directory):
        return docs

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            path = os.path.join(directory, filename)

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                rows = list(reader)
                text = "\n".join([", ".join(row) for row in rows])
                docs.append(Document(text=text))
    return docs


def build_pdf_index():
    print("Loading PDF documents...")
    pdf_docs = SimpleDirectoryReader(PDF_DIR, recursive=True).load_data()

    print("Loading CSV files...")
    csv_docs = load_csv_files(CSV_DIR)

    all_docs = pdf_docs + csv_docs

    print(f"Total documents loaded: {len(all_docs)}")

    embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    index = VectorStoreIndex.from_documents(
        all_docs,
        embed_model=embed_model
    )

    index.storage_context.persist(persist_dir=PDF_INDEX_DIR)
    print("PDF index built and saved.")


def load_pdf_index():
    storage_context = StorageContext.from_defaults(persist_dir=PDF_INDEX_DIR)
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    return load_index_from_storage(storage_context, embed_model=embed_model)


# ============================================================
# MEMORY-AWARE HYBRID RAG QUERY
# ============================================================
def ask_rag(question: str):
    pdf_index = load_pdf_index()
    email_index = load_email_index()

    llm = Ollama(model="gemma3:1b", request_timeout=300.0)
    memory = load_memory()

    # Memory context
    history_text = "\n".join(
        [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in memory["conversation_history"][-10:]]
    )
    prefs_text = "\n".join([f"{k}: {v}" for k, v in memory["user_preferences"].items()])
    facts_text = "\n".join(memory["learned_facts"])

    memory_context = f"""
    ### Conversation Memory ###
    {history_text}

    ### User Preferences ###
    {prefs_text}

    ### Learned Facts ###
    {facts_text}
    """

    # PDF retrieval
    pdf_engine = pdf_index.as_query_engine(
        llm=llm,
        similarity_top_k=4,
        response_mode="compact"
    )
    pdf_result = pdf_engine.query(question)

    # Email retrieval
    email_engine = email_index.as_query_engine(
        llm=llm,
        similarity_top_k=4,
        response_mode="compact"
    )
    email_result = email_engine.query(question)

    # Final combined prompt
    final_prompt = f"""
    {memory_context}

    ### Relevant PDF Context ###
    {pdf_result}

    ### Relevant Email Context ###
    {email_result}

    ### User Question ###
    {question}

    Use all relevant context above to answer accurately.
    """

    final_answer = llm(final_prompt)
    remember_conversation(question, str(final_answer))

    return final_answer


# ============================================================
# MAIN LOOP
# ============================================================
if __name__ == "__main__":
    print("Building PDF index...")
    build_pdf_index()

    print("Building Email index...")
    build_email_index()

    print("\nHybrid Agentic RAG system ready. Ask questions (type 'exit' to quit).")

    while True:
        q = input("\nYour question: ").strip()
        if q.lower() in ["exit", "quit"]:
            break

        # Memory commands
        if "remember that" in q.lower():
            fact = q.split("remember that", 1)[1].strip()
            remember_fact(fact)
            print("Got it. I will remember that.")
            continue

        if "my preference is" in q.lower():
            pref = q.split("my preference is", 1)[1].strip()
            key, value = pref.split("=", 1)
            remember_preference(key.strip(), value.strip())
            print("Preference saved.")
            continue

        # Hybrid RAG
        answer = ask_rag(q)
        print("\nAnswer:", answer)

