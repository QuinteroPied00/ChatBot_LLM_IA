# 📌 Curso Práctico: Construcción de un Chatbot con IA Generativa y Base de Datos Vectorial

Este curso guiará a los estudiantes en la creación de un **chatbot con IA generativa** utilizando **LLaMA, LangChain, ChromaDB y Streamlit**. Se abordará desde la instalación y configuración del entorno hasta la optimización de respuestas a través de técnicas avanzadas de **ingeniería de prompts**.

---

## **1️⃣ Introducción a la IA Generativa y Bases de Datos Vectoriales**

### **📌 Inteligencia Artificial Generativa**   

La IA generativa hace referencia a modelos capaces de **generar texto, imágenes, código y otros tipos de contenido** basado en patrones aprendidos a partir de grandes volúmenes de datos. Modelos como **LLaMA** utilizan arquitecturas de **Transformers**, que permiten procesar información secuencialmente, atendiendo relaciones contextuales dentro del texto.

### **📌 Bases de Datos Vectoriales**

Las bases de datos vectoriales, como **ChromaDB**, permiten almacenar información en forma de **vectores embebidos**. Esto facilita la búsqueda y recuperación eficiente de información basada en **similitud semántica**, en lugar de coincidencias exactas de palabras clave.

📍 **Cómo lo implementamos en este curso:**

- mejoraUtilizamos **HuggingFaceEmbeddings** para convertir el texto en representaciones vectoriales.
- Almacenamos estos vectores en **ChromaDB** para facilitar consultas en lenguaje natural.
- Integramos estas consultas en **LangChain**, combinando bases de datos y modelos generativos para generar respuestas relevantes.

---

## **2️⃣ Instalación del Entorno de Desarrollo**

### **🔹 Paso 1: Crear y Activar un Entorno Virtual**

```bash
# Crear un entorno virtual
python3 -m venv venv

# Activar el entorno virtual (Linux/macOS)
source venv/bin/activate

# Activar el entorno virtual (Windows)
venv\Scripts\activate
```

### **🔹 Paso 2: Instalar las Dependencias**

```bash
pip install streamlit langchain langchain-community ollama chromadb sentence-transformers
```

---

## **3️⃣ Configuración del Logger**

Para facilitar la depuración y monitoreo del chatbot, configuramos un sistema de **logging** que capture eventos clave en la ejecución.

📍 **Archivo ****`logger.py`**:

```python
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

---

## **4️⃣ Implementación de la Base de Datos Vectorial**

📍 **Archivo ****`database.py`**:

```python
import chromadb
from modules.logger import logger
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_vectorstore():
    """
    Inicializa y devuelve un almacén de vectores utilizando ChromaDB.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    except Exception as e:
        logger.error(f"Error al conectar con la base de datos vectorial: {e}")
        return None
```

---

## **5️⃣ Carga y Almacenamiento de Documentos**

📍 **Archivo ****`embeddings.py`**:

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

## **6️⃣ Implementación del Chatbot**

📍 **Archivo ****`chatbot.py`**:

```python
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from modules.database import get_vectorstore
from modules.logger import logger

def chat_with_llama(user_input, vectorstore):
    """
    Genera una respuesta utilizando el modelo LLaMA con la información de ChromaDB.
    """
    try:
        llm = OllamaLLM(model="llama2")
        memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
        retriever = vectorstore.as_retriever() if vectorstore else None
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        
        response = retrieval_chain.run(user_input) if retrieval_chain else "No se pudo generar respuesta."
        return response
    except Exception as e:
        logger.error(f"Error en la generación de respuesta del chatbot: {e}")
        return "Lo siento, hubo un error al procesar tu solicitud."
```

---

## **7️⃣ Construcción de la Aplicación con Streamlit**

📍 **Archivo ****`app.py`**:

📍 **Archivo ****`app.py`**:

```python
import streamlit as st
from modules.chatbot import chat_with_llama
from modules.embeddings import load_and_store_documents
from modules.logger import logger
from modules.database import get_vectorstore

# Configuración de la interfaz de Streamlit
st.set_page_config(page_title="Chat con LLaMA y Base de Datos", layout="wide")
st.title("🤖 Chat con LLaMA y Base de Datos Vectorial")

# Cargar la base de datos vectorial una sola vez
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_vectorstore()
    logger.info("Base de datos vectorial cargada en memoria.")

# Sección para cargar documentos desde la interfaz
st.sidebar.header("Carga de Documentos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo de texto o datos", type=["txt", "csv", "json"])

if uploaded_file is not None:
    ruta_guardado = f"data/documentos/{uploaded_file.name}"
    with open(ruta_guardado, "wb") as f:
        f.write(uploaded_file.getbuffer())
    load_and_store_documents(uploaded_file, uploaded_file.name)
    logger.info(f"Archivo {uploaded_file.name} cargado e indexado correctamente.")
    st.sidebar.success(f"Archivo {uploaded_file.name} cargado correctamente y agregado al contexto.")

# Inicializar el historial de conversación
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial de conversación
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada del usuario
user_input = st.chat_input("Escribe tu mensaje...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Utilizar la base de datos vectorial precargada
    response = chat_with_llama(user_input, st.session_state.vectorstore)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
```

---

## **8️⃣ Ingeniería de Prompts**

## **8️⃣ Ingeniería de Prompts para Optimizar el Chatbot**

La **ingeniería de prompts** es una técnica clave en la interacción con modelos de lenguaje natural. Consiste en diseñar instrucciones precisas y estructuradas para obtener respuestas más relevantes y contextuales del chatbot.

### **📌 Fundamentos de la Ingeniería de Prompts**

Los modelos de lenguaje, como LLaMA, generan respuestas basándose en patrones lingüísticos y contextos aprendidos. La calidad de la respuesta está directamente relacionada con la claridad, precisión y contexto del prompt ingresado.

### **📌 Tipos de Prompts y su Implementación en el Chatbot**

#### **🔹 Preguntas Directas**

Este tipo de preguntas buscan respuestas específicas y concretas.

✅ **Ejemplo:**

```text
¿Qué ciudad menciona User 1 en la conversación sobre mudanza?
```

📍 **Cómo funciona en el chatbot:** El modelo buscará en la base de datos vectorial y extraerá la ciudad mencionada en el diálogo.

#### **🔹 Prompts con Contexto Expandido**

Para mejorar la precisión, podemos incluir más información en el prompt.

✅ **Ejemplo:**

```text
User 1 está emocionado por mudarse a una nueva ciudad para seguir su sueño. ¿A qué ciudad se refiere y por qué?
```

📍 **Ventaja:** Se guía al modelo a procesar información dentro de un marco más definido.

#### **🔹 Prompts Comparativos**

Se usan para comparar información entre distintos fragmentos del contexto.

✅ **Ejemplo:**

```text
¿Quién tiene más afinidad con la naturaleza, User 1 o User 2?
```

📍 **Cómo funciona en el chatbot:** Permite analizar distintas interacciones para evaluar quién menciona más actividades al aire libre.

#### **🔹 Preguntas sobre Emociones y Motivaciones**

Este tipo de preguntas buscan inferir estados emocionales y razones detrás de una acción en el diálogo.

✅ **Ejemplo:**

```text
¿Qué motiva a User 2 a ser bombero y mudarse de ciudad?
```

📍 **Cómo funciona en el chatbot:** Se extraerán indicios emocionales en el texto para justificar su decisión.

#### **🔹 Reformulación de Preguntas Ambiguas**

Si el chatbot no responde de forma precisa, podemos hacer preguntas más guiadas.

❌ **Pregunta Ambigua:**

```text
¿Qué sabe el chatbot sobre Portland?
```

✅ **Mejor Pregunta:**

```text
¿Qué mencionan los personajes en el contexto sobre lugares emblemáticos en Portland?
```

📍 **Cómo funciona en el chatbot:** Reduce la ambigüedad y guía mejor la búsqueda en la base de datos vectorial.

### **📌 Implementación en Streamlit**

Podemos agregar una opción en `app.py` para que los usuarios seleccionen estrategias de ingeniería de prompts.

```python
st.sidebar.header("Estrategia de Ingeniería de Prompts")
prompt_type = st.sidebar.selectbox("Selecciona un tipo de prompt", [
    "Pregunta Directa",
    "Contexto Expandido",
    "Comparación",
    "Emociones y Motivaciones",
    "Reformulación"
])
```

### **📌 Consejos para Crear Prompts Más Efectivos**

1️⃣ **Sé específico:** Cuanto más claro sea el prompt, mejor será la respuesta.
2️⃣ **Añade contexto adicional:** Explicar el propósito de la pregunta mejora la precisión.
3️⃣ **Reformula si es necesario:** Si la respuesta es imprecisa, intenta una variación del prompt.
4️⃣ **Evita preguntas demasiado generales:** Preguntas vagas pueden generar respuestas poco informativas.
5️⃣ **Usa ejemplos dentro del prompt:** Si es necesario, incluye referencias en la instrucción.

---

## **🎯 Resultados Esperados**

✅ **Mejor precisión en las respuestas del chatbot.**\
✅ **Capacidad de obtener respuestas más relevantes del contexto.**\
✅ **Mayor control sobre cómo el chatbot interactúa con los datos almacenados.**

🚀 **Con estas técnicas, optimizamos la calidad de las respuestas del chatbot y aprovechamos al máximo la base de datos vectorial!** 🎯

---

## **🎯 Resultados Esperados**

✅ Un chatbot interactivo que puede responder preguntas sobre el contexto cargado.\
✅ Integración con **ChromaDB** para mejorar la precisión de las respuestas.\
✅ Optimización mediante **ingeniería de prompts** para mejorar la calidad de la generación de texto.



