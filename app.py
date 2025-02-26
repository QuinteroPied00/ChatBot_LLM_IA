import streamlit as st
from modules.chatbot import chat_with_llama
from modules.embeddings import load_and_store_documents
from modules.logger import logger

st.set_page_config(page_title="Chat con LLaMA y Base de Datos", layout="wide")
st.title("🤖 Chat con LLaMA y Base de Datos Vectorial")

st.sidebar.header("Carga de Documentos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo de texto o datos", type=["txt", "csv", "json"])
if uploaded_file is not None:
    ruta_guardado = f"data/documentos/{uploaded_file.name}"
    with open(ruta_guardado, "wb") as f:
        f.write(uploaded_file.getbuffer())
    load_and_store_documents(uploaded_file, uploaded_file.name)
    logger.info(f"Archivo {uploaded_file.name} cargado e indexado correctamente.")
    st.sidebar.success(f"Archivo {uploaded_file.name} cargado correctamente y agregado al contexto.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Escribe tu mensaje...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    response = chat_with_llama(user_input)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)