# 📌 Curso Práctico: Construcción de un Chatbot con IA Generativa y Base de Datos Vectorial

Este curso guiará a los estudiantes paso a paso en la creación de un **chatbot con IA generativa** usando **LLaMA, LangChain, ChromaDB y Streamlit**.

---

## **Conceptos Clave Implementados en el Chatbot**

### **📌 1. Embeddings** 
Los **embeddings** son representaciones vectoriales de palabras o frases que permiten que el chatbot entienda relaciones semánticas entre términos. En nuestro chatbot, los embeddings se generan con **HuggingFaceEmbeddings** y se almacenan en ChromaDB para ser utilizados en la búsqueda de información relevante.

📍 **Dónde se usa en el código:**
- En `database.py`, la función `get_vectorstore()` genera embeddings usando el modelo `sentence-transformers/all-MiniLM-L6-v2`.
- En `embeddings.py`, los documentos cargados se dividen en fragmentos y se convierten en embeddings antes de almacenarlos en la base de datos.

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore.add_documents(docs)
```

### **📌 2. Modelos Transformers**
Los **transformers** son modelos de IA especializados en el procesamiento del lenguaje natural (NLP). Se basan en la arquitectura de atención para comprender el contexto de las palabras en una oración. En este proyecto, utilizamos **LLaMA**, un modelo basado en transformers, para generar respuestas contextuales.

📍 **Dónde se usa en el código:**
- En `chatbot.py`, el modelo **OllamaLLM** es el encargado de generar respuestas basadas en la consulta del usuario.

```python
llm = OllamaLLM(model="llama2")
```

### **📌 3. Cadenas (Chains) en LangChain**
Las **cadenas (Chains)** en LangChain permiten conectar diferentes componentes, como modelos de lenguaje y bases de datos vectoriales, para construir flujos de trabajo complejos. En nuestro chatbot, utilizamos una cadena de recuperación conversacional (`ConversationalRetrievalChain`) para responder a preguntas basándose en el contexto almacenado.

📍 **Dónde se usa en el código:**
- En `chatbot.py`, la cadena `ConversationalRetrievalChain` se configura para recuperar información relevante antes de generar la respuesta.

```python
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)
```

### **📌 4. Base de Datos Vectorial (ChromaDB)**
Las bases de datos vectoriales almacenan información en forma de embeddings para permitir una búsqueda rápida y eficiente. En este chatbot, utilizamos **ChromaDB** para indexar y recuperar documentos relevantes.

📍 **Dónde se usa en el código:**
- En `database.py`, `get_vectorstore()` inicializa y gestiona la base de datos vectorial.
- En `embeddings.py`, los documentos se convierten en embeddings y se almacenan en ChromaDB.
- En `chatbot.py`, `get_vectorstore().as_retriever()` permite la búsqueda en la base de datos.

```python
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever()
```

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

## **2️⃣ Configuración del Logger**

Creamos un archivo `logger.py` dentro de la carpeta `modules/` con el siguiente código:

```python
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

---

## **3️⃣ Configuración de la Base de Datos Vectorial**

Creamos un archivo `database.py` dentro de la carpeta `modules/` con el siguiente código:

```python
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
```

---

## **4️⃣ Cargar Documentos en la Base de Datos**

Creamos un archivo `embeddings.py` dentro de la carpeta `modules/` con el siguiente código:

```python
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
```

---

## **5️⃣ Implementar el Chatbot y Almacenar el Historial**

Creamos un archivo `chatbot.py` dentro de la carpeta `modules/` con el siguiente código:

```python
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
        llm = OllamaLLM(model="llama2")
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
```

---

## **6️⃣ Construcción de la Aplicación con Streamlit**

Creamos `app.py` en la raíz del proyecto con la siguiente configuración:

```python
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
```

---

## **7️⃣ Ejecutar el Proyecto**

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

✅ Un chatbot interactivo que puede responder preguntas.
✅ Capacidad de **cargar documentos** y usarlos como contexto.
✅ Integración con **ChromaDB** para mejorar la precisión de las respuestas.
✅ **Historial de búsqueda almacenado** para consultar interacciones previas.


