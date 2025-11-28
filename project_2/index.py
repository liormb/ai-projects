import os
from dotenv import load_dotenv
from utils.helpers import (
    print_app_info,
    is_faiss_exists,
    get_sources_path,
    load_documents,
    chunk_documents,
    get_embeddings_model,
    create_vector_store,
    get_retriever,
    test_chatbot,
)
load_dotenv(".env")

print_app_info()

chat_history = []
faiss_path = os.getenv("FAISS_INDEX_PATH")
remove_cache = os.getenv("REMOVE_FAISS_ON_STARTUP").lower() == "true"
has_faiss_store = is_faiss_exists(faiss_path, remove_cache)

if not has_faiss_store:
    sources = get_sources_path()
    docs = load_documents(sources)
    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=30)
    embeddings = get_embeddings_model(chunks)
    create_vector_store(chunks, embeddings)

retriever = get_retriever(faiss_path, k=3)
test_chatbot(retriever, chat_history)
