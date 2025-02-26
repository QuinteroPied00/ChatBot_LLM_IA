# 📌 Curso Práctico: Construcción de un Chatbot con IA Generativa y Base de Datos Vectorial

Este curso guiará a los estudiantes paso a paso en la creación de un **chatbot con IA generativa** usando **LLaMA, LangChain, ChromaDB y Streamlit**.

---

## **1️⃣ Instalación de Dependencias**

### **🔹 Paso 1: Crear y Activar un Entorno Virtual**

Ejecuta los siguientes comandos en la terminal:

```bash
# Crear un entorno virtual
python3 -m venv venv

# Activar el entorno virtual (Linux/macOS)
source venv/bin/activate

# Activar el entorno virtual (Windows)
venv\Scripts\activate
```

### **🔹 Paso 2: Instalar las Dependencias**

Ejecuta:

```bash
pip install streamlit langchain langchain-community ollama chromadb sentence-transformers
```

---

## **2️⃣ Configuración de la Base de Datos Vectorial**

Creamos un archivo `database.py` dentro de la carpeta `modules/` con el siguiente código:

```python
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Inicializar la base de datos vectorial
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="chat_docs")

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
```

---

## **3️⃣ Cargar Documentos en la Base de Datos**

Creamos `embeddings.py` dentro de `modules/` para procesar documentos subidos en Streamlit:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from modules.database import get_vectorstore

def load_and_store_documents(file_content, file_name):
    """Carga documentos desde Streamlit, genera embeddings y los almacena en ChromaDB"""
    with open(f"data/documentos/{file_name}", "wb") as f:
        f.write(file_content.getbuffer())
    
    loader = TextLoader(f"data/documentos/{file_name}")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    vectorstore = get_vectorstore()
    vectorstore.add_documents(docs)
    vectorstore.persist()
    
    print("Documentos indexados correctamente.")
```

---

## **4️⃣ Implementar el Chatbot y Almacenar el Historial**

Creamos `chatbot.py` dentro de `modules/`:

```python
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from modules.database import get_vectorstore
import json
import os

HISTORY_FILE = "data/historial/search_history.json"

def ensure_history_file():
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)

def save_search(query, response):
    ensure_history_file()
    with open(HISTORY_FILE, "r+") as f:
        history = json.load(f)
        history.append({"query": query, "response": response})
        f.seek(0)
        json.dump(history, f, indent=4)

def chat_with_llama(user_input):
    llm = Ollama(model="llama2")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=get_vectorstore().as_retriever(),
        memory=memory
    )
    response = retrieval_chain.run(user_input)
    save_search(user_input, response)
    return response
```

---

## **5️⃣ Construcción de la Aplicación en Streamlit**

Creamos `app.py` en la raíz del proyecto con la siguiente configuración:

```python
import streamlit as st
from modules.chatbot import chat_with_llama
from modules.embeddings import load_and_store_documents

st.set_page_config(page_title="Chat con LLaMA y Base de Datos", layout="wide")
st.title("🤖 Chat con LLaMA y Base de Datos Vectorial")

st.sidebar.header("Carga de Documentos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo de texto", type=["txt"])
if uploaded_file is not None:
    load_and_store_documents(uploaded_file, uploaded_file.name)
    st.sidebar.success(f"Archivo {uploaded_file.name} cargado y agregado al contexto.")

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
```

---

## **6️⃣ Ejecutar el Proyecto**

### **🔹 Paso 1: Iniciar Ollama**

```bash
ollama serve
```

### **🔹 Paso 2: Ejecutar Streamlit**

```bash
streamlit run app.py
```

---

## **🎯 Resultados Esperados**

✅ Un chatbot interactivo que puede responder preguntas. ✅ Capacidad de **cargar documentos** y usarlos como contexto. ✅ Integración con **ChromaDB** para mejorar la precisión de las respuestas. ✅ **Historial de búsqueda almacenado** para consultar interacciones previas.

---


