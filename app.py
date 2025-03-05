import streamlit as st
from modules.chatbot import chat_with_llama
from modules.embeddings import load_and_store_documents
from modules.database import get_vectorstore
from modules.logger import logger
import os

st.set_page_config(page_title="Chat con LLaMA y Base de Datos", layout="wide")
st.title("Chismosea con Karol G")

st.sidebar.header("Carga de Documentos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo de texto o datos", type=["txt", "csv", "json"])

if uploaded_file is not None:
    # Verifica si ya se ha cargado y almacenado antes
    file_name = uploaded_file.name
    file_path = f"data/documentos/{file_name}"

    if not os.path.exists(file_path):  # Si no existe, significa que no se ha indexado
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        load_and_store_documents(uploaded_file, file_name)
        logger.info(f"Archivo {file_name} cargado e indexado correctamente.")
        st.sidebar.success(f"Archivo {file_name} cargado correctamente y agregado al contexto.")
    else:
        # Si el archivo ya está cargado e indexado
        logger.info(f"El archivo {file_name} ya ha sido indexado previamente.")
        st.sidebar.info(f"El archivo {file_name} ya ha sido indexado previamente.") #
        


if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.header("Estrategia de Ingeniería de Prompts")
prompt_type = st.sidebar.selectbox("Selecciona un tipo de prompt", [
    "Pregunta Directa",
    "Contexto Expandido",
    "Comparación",
    "Emociones y Motivaciones",
    "Reformulación"
])

# Mostrar historial de conversación
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


    
    # Aplicar estrategia de prompt engineering
    user_input = ""
    if prompt_type == "Contexto Expandido":
        user_input = f"Basado en el contexto, responde lo siguiente: {user_input}"
    elif prompt_type == "Comparación":
        user_input = f"Compara la información disponible y responde: {user_input}"
    elif prompt_type == "Emociones y Motivaciones":
        user_input = f"Analiza las emociones detrás del mensaje y responde: {user_input}"
    elif prompt_type == "Reformulación":
        user_input = f"Si la pregunta no es clara, reformula y responde: {user_input}"

# Entrada del usuario
user_input = st.chat_input("Escribe tu mensaje...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Verificar si la base de datos vectorial está cargada
    if "vectorstore" in st.session_state and st.session_state.vectorstore:
        response = chat_with_llama(user_input, st.session_state.vectorstore)
    else:
        response = "Error: La base de datos vectorial no está disponible. Intentando recargar..."
        logger.error("Intento de consulta sin base de datos vectorial cargada. Se intentará una nueva inicialización.")
        st.session_state.vectorstore = get_vectorstore()
        if st.session_state.vectorstore:
            response = chat_with_llama(user_input, st.session_state.vectorstore)
        else:
            response = "Error crítico: No se pudo recuperar la base de datos vectorial."
            st.error("Error crítico: No se pudo recuperar la base de datos vectorial. Verifique la conexión con ChromaDB.")
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
