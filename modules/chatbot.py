from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from modules.database import get_vectorstore
from modules.logger import logger

# Almacenar el historial de conversaciones
def save_conversation(user_input, response):
    """
    Guarda la conversación en un archivo de historial.
    """
    try:
        with open("conversation_history.txt", "a") as file:
            file.write(f"Usuario: {user_input}\n")
            file.write(f"Chatbot: {response}\n\n")
    except Exception as e:
        logger.error(f"Error al guardar el historial de la conversación: {e}")


def chat_with_llama(user_input, vectorstore):
    """
    Genera una respuesta utilizando el modelo LLaMA con la información de ChromaDB.
    """
    try:
        llm = OllamaLLM(model="deepseek-r1:1.5b")
        memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
        retriever = vectorstore.as_retriever() if vectorstore else None
        
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )

        response = retrieval_chain.invoke(user_input) if retrieval_chain else "No se pudo generar respuesta."
        logger.info(f"Respuesta generada: {response['answer']}")
        
        # Guardar la conversación en el historial
        save_conversation(user_input, response['answer'])
        
        return response['answer']
    except Exception as e:
        logger.error(f"Error en la generación de respuesta del chatbot: {e}")
        return "Lo siento, hubo un error al procesar tu solicitud."
