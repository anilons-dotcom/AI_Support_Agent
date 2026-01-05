from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document  # Optional, for type hinting

# Paths (match these to your build script)
VECTOR_STORE_PATH = "fixed_income_index.faiss"

# Load the vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

# Sample queries (customize based on your PDF content, e.g., fixed income topics like bonds)
queries = [
    "tell what all information is with in the pdf ",
    "Explain anil back grounds",
    "How does  consent form works?"
]

# Test similarity search
for query in queries:
    print(f"\nQuery: {query}")
    results = vectorstore.similarity_search(query, k=3)  # Retrieve top 3 matches
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Content: {doc.page_content[:200]}...")  # Truncate for brevity
        print(f"Metadata: {doc.metadata}")
        print("---")

print("\nTest complete. If results are relevant, the knowledge base is working!")