from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from modules.database import get_vectorstore
from modules.logger import logger

def load_and_store_documents(file_content, file_name):
    """
    Carga documentos subidos a través de Streamlit, genera embeddings y los almacena en ChromaDB.
    """
    try:
        with open(f"data/documentos/{file_name}", "wb") as f:
            f.write(file_content.getbuffer())
        
        loader = TextLoader(f"data/documentos/{file_name}")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        vectorstore = get_vectorstore()
        if vectorstore:
            vectorstore.add_documents(docs)
            logger.info(f"Documentos del archivo {file_name} indexados correctamente en ChromaDB.")
        else:
            logger.warning("No se pudo indexar el documento debido a un problema con la base de datos vectorial.")
    except Exception as e:
        logger.error(f"Error al cargar y almacenar documentos: {e}")