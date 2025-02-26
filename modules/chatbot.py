from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from modules.database import get_vectorstore
from modules.logger import logger
import json
import os

HISTORY_FILE = "data/historial/search_history.json"

def ensure_history_file():
    """
    Verifica la existencia del archivo de historial y lo crea si no existe.
    """
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)

def save_search(query, response):
    """
    Guarda una consulta y su respuesta en el historial de búsquedas.
    """
    ensure_history_file()
    with open(HISTORY_FILE, "r+") as f:
        history = json.load(f)
        history.append({"query": query, "response": response})
        f.seek(0)
        json.dump(history, f, indent=4)

def chat_with_llama(user_input):
    """
    Genera una respuesta a partir de una consulta utilizando el modelo LLaMA.
    """
    try:
        llm = OllamaLLM(model="deepseek-r1:1.5b")
        memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever() if vectorstore else None
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        
        response = retrieval_chain.run(user_input) if retrieval_chain else "Error: No se pudo generar una respuesta debido a problemas con el contexto."
        save_search(user_input, response)
        return response
    except Exception as e:
        logger.error(f"Error en la generación de respuesta del chatbot: {e}")
        return "Lo siento, hubo un error al procesar tu solicitud."
