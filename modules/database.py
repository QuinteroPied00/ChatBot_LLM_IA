import chromadb
from modules.logger import logger
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_vectorstore():
    """
    Inicializa y devuelve un almacén de vectores utilizando ChromaDB.
    Utiliza HuggingFaceEmbeddings para generar los embeddings de los documentos.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    except Exception as e:
        logger.error(f"Error al conectar con la base de datos vectorial: {e}")
        return None