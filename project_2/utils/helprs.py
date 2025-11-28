import os, glob, pathlib, random
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

root = str(Path(__file__).resolve().parents[1])
env_path = os.path.join(root, ".env")

load_dotenv(env_path)

def print_app_info():
    print("""
         .-.
        ( o )
    .---(   )---.
    |  CHATBOT  |
    '-----------'
    """)

def print_char(char="-", length=50):
    print(char * length)

# --------------------------------------------------------------
# Check if FAISS vector store exists
# --------------------------------------------------------------
def is_faiss_exists(path="faiss_index", remove_cache=False):
    if remove_cache:
        delete_vector_store(path)
        print("🗑️ Removed existing FAISS vector store and cache.")
    return pathlib.Path(path).exists()

# --------------------------------------------------------------
# Get sources paths (PDFs and URLs)
# --------------------------------------------------------------
def get_sources_path():
    path = os.path.join(root, "data/Everstorm_*.pdf")
    PDFs = glob.glob(path)
    URLs = [
        "https://developer.bigcommerce.com/docs/store-operations/shipping",
        "https://developer.bigcommerce.com/docs/store-operations/orders/refunds",
        "https://docs.stripe.com/disputes",
        "https://woocommerce.github.io/woocommerce-rest-api-docs/v3.html",
    ]
    return [*PDFs, *URLs]

# --------------------------------------------------------------
# Load documents from various sources
# --------------------------------------------------------------
def load_documents(sources=[]):
    docs = []
    url_identifier = "http"

    for source in sources:
        try:
            is_url = source.startswith(url_identifier)
            loader = UnstructuredURLLoader(urls=[source]) if is_url else PyPDFLoader(source)
            docs.extend(loader.load())
        except Exception as e:
            print(f"⚠️ Failed to load {source}: {e}")

    print(f"📄 Loaded {len(docs)} documents from sources.")
    return docs

# --------------------------------------------------------------
# Split documents into smaller chunks for embedding
# --------------------------------------------------------------
def chunk_documents(docs=[], chunk_size=300, chunk_overlap=30):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    for doc in docs:
        doc_chunks = splitter.split_text(doc.page_content)
        chunks.extend(doc_chunks)

    print(f"✂️  Split documents into {len(chunks)} chunks.")
    return chunks

# --------------------------------------------------------------
# Get cached embeddings if available
# --------------------------------------------------------------
def get_cached_embeddings(path=None):
    return np.load(path) if pathlib.Path(path).exists() else None

# --------------------------------------------------------------
# Generate embeddings for chunks (vector-based indexing)
# --------------------------------------------------------------
def generate_embeddings(chunks=[], path=None):
    client = OpenAI()
    embeddings_model = os.getenv("EMBEDDING_MODEL_NAME")
    response = client.embeddings.create(
        model=embeddings_model,
        input=chunks,
        encoding_format="float"
    )
    embeddings_vectors = np.array([data_point.embedding for data_point in response.data]) # Vector array
    np.save(path, embeddings_vectors)

    print(f"🧠 Generated and cached embeddings: {embeddings_vectors.shape[0]} vectors, each with {embeddings_vectors.shape[1]} dimensions.")
    return embeddings_vectors
# --------------------------------------------------------------
# Get embeddings model (with caching)
# --------------------------------------------------------------
def get_embeddings_model(chunks=[], path=None):
    path = path or pathlib.Path(os.getenv("EMBEDDING_CACHE_PATH"))
    cached = get_cached_embeddings(path)

    if cached is not None:
        print(f"🧠 Loaded {len(cached)} cached embeddings.")
        return cached
    return generate_embeddings(chunks, path)

# --------------------------------------------------------------
# Delete embedding cache file
# --------------------------------------------------------------
def delete_embedding_cache(path=None):
    path = path or pathlib.Path(os.getenv("EMBEDDING_CACHE_PATH"))
    if pathlib.Path(path).exists():
        os.remove(path)
        print("🗑️ Deleted embedding cache.")

# --------------------------------------------------------------
# Create and save FAISS vector store
# --------------------------------------------------------------
def create_vector_store(chunks=[], embeddings=[], path="faiss_index"):
    data = list(zip(chunks, embeddings))
    vectordb = FAISS.from_embeddings(data, OpenAIEmbeddings()) # Create FAISS vector store
    vectordb.save_local(Path(path).name)
    print(f"💾 Created and saved FAISS vector store with {len(chunks)} vectors.")

# --------------------------------------------------------------
# Load FAISS vector store from disk
# --------------------------------------------------------------
def load_vector_store(path="faiss_index", k=8):
    store = FAISS.load_local(
        Path(path).name,
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )
    print(f"💾 Loaded FAISS vector store from {Path(path).name}.")
    return store.as_retriever(search_kwargs={"k": k}) # Return retriever with k=8

# --------------------------------------------------------------
# Delete FAISS vector store
# --------------------------------------------------------------
def delete_vector_store(path="faiss_index"):
    dir_path = Path(path).name

    if pathlib.Path(dir_path).exists():
        for file in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, file))
        os.rmdir(dir_path)
        print("🗑️ Deleted FAISS vector store.")
        delete_embedding_cache() # Also delete embedding cache

# --------------------------------------------------------------
# Get retriever from existing FAISS vector store
# --------------------------------------------------------------
def get_retriever(path="faiss_index", k=None):
    path = Path(path).name
    k = k or random.choice([3, 5, 8, 10])
    retriever = load_vector_store(path, k)

    if retriever is None:
        raise ValueError("❌ Failed to load FAISS vector store retriever.")

    print(f"🔍 Retrieving {k} relevant documents per question.")
    return retriever

# --------------------------------------------------------------
# Ask a question using the retriever and LLM
# --------------------------------------------------------------
def ask_question(retriever, question, history=[], temperature=0.1):
    llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL_NAME"), temperature=temperature)
    relevant_docs = retriever.invoke(question)

    prompt = ChatPromptTemplate.from_template(
        template=os.getenv("PROMPT_TEMPLATE")
    ).format_prompt(
        system=os.getenv("SYSTEM_PROMPT"),
        context="".join([doc.page_content for doc in relevant_docs]),
        question=question,
        chat_history="\n".join(history),
    ).to_string()

    return llm.invoke(prompt).strip()

# --------------------------------------------------------------
# Test the chatbot with predefined questions
# --------------------------------------------------------------
def test_chatbot(retriever, history=[]):
    questions = [
        "If I'm not happy with my purchase, what is your refund policy and how do I start a return?",
        "How long will delivery take for a standard order, and where can I track my package once it ships?",
        "What's the quickest way to contact your support team, and what are your operating hours?",
    ]

    for question in questions:
        print_char('-')
        answer = ask_question(retriever, question, history)
        history.append(f"Q: {question}\nA: {answer}\n")
        print("💬 Question:", question)
        print("💬 Answer:", answer)

    print_char("=")
    print(f"✅ Completed testing chatbot with {len(questions)} questions.\n")