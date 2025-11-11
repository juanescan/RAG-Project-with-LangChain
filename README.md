# RAG-Project-with-LangChain

Este proyecto implementa un sistema de **Retrieval Augmented Generation (RAG)** utilizando **LangChain** y **Pinecone** en **Google Colab**.  
El objetivo es crear una aplicación que pueda responder preguntas basadas en el contenido de un sitio web, utilizando un modelo de lenguaje (LLM) para generar respuestas y un almacén de vectores para recuperar información relevante.

---

## **Descripción del Proyecto**

Este proyecto implementa un sistema RAG que:

1. **Indexa documentos**: Carga, divide y almacena documentos en una base de datos vectorial (Pinecone).  
2. **Recupera información relevante**: Dada una pregunta, recupera los fragmentos de texto más relevantes.  
3. **Genera respuestas**: Usa un modelo de lenguaje (GPT-4) para generar respuestas basadas en la información recuperada.

---

## **Arquitectura y Componentes**

El proyecto está compuesto por los siguientes componentes principales:

1. **LangChain**: Framework para construir aplicaciones con modelos de lenguaje.  
2. **OpenAI**: Proveedor del modelo de lenguaje (GPT-4) y embeddings.  
3. **Pinecone**: Base de datos vectorial para almacenar y recuperar embeddings de documentos.  
4. **BeautifulSoup4**: Biblioteca para analizar y extraer contenido de páginas web.  
5. **LangGraph**: Framework para construir flujos de trabajo complejos con modelos de lenguaje.

---

## **Requisitos**

- Python 3.8+
- Cuenta en [OpenAI](https://platform.openai.com/) para obtener una clave API.
- Cuenta en [Pinecone](https://www.pinecone.io/) para usar la base de datos vectorial.

---

## **Instalación de Dependencias**

Ejecuta los siguientes comandos para instalar las bibliotecas necesarias:

```bash
!pip install langchain openai langchain-openai langchain-community langchain-text-splitters langchainhub pinecone-client
!pip install -qU langchain-openai
!pip install -qU langchain-pinecone pinecone-notebooks
!pip install langgraph
!pip install bs4
```

LangChain-RAG-AREP/
├── README.md
├── src/
│   └── rag-project.ipynb

README.md: Documentación del proyecto.
rag-project.ipynb: Código principal del sistema RAG.

## 🔑 Configuración del Entorno

### 🧰 Configuración de OpenAI

Agrega tu API Key de OpenAI al entorno:

```bash
import getpass, os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

```

## 🪣 Configuración de Pinecone
Configura la API Key de Pinecone:

```bash
import getpass, os

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

```

## 🧠 Componentes del Proyecto

### 🔤 Modelo de Lenguaje (LLM)

```bash
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4", model_provider="openai")
```

### 🧩 Modelo de Embeddings

```bash
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

### 📦 Almacén de Vectores (Pinecone)

```bash
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_api_key)
index_name = "langchain-test-index"

existing_indexes = [i["name"] for i in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
```

### 🌐 Carga y División de Documentos

```bash
import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))},
)
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
```

### 🧾 Indexación de Documentos

```bash
from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index=index, embedding=embeddings)
_ = vector_store.add_documents(documents=all_splits)
```

### 🕸️ Construcción del Grafo con LangGraph

```bash
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
```

### 💬 Ejecución del Proyecto

```bash
response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])
```

### 🧭 Uso del Proyecto

1. Configura tus claves de OpenAI y Pinecone.
2. Ejecuta las celdas del notebook en orden.
3. Realiza preguntas con graph.invoke({"question": "tu pregunta"}).

### 🕸️ Visualización del Grafo
Puedes visualizar el grafo de LangGraph para entender el flujo de datos:

![](/imagenes/1.png)


## 👨‍💻 Autor

- Juan Esteban Cancelado - *AREP* *LangChain-LLM-AREP* - [juanescan](https://github.com/juanescan)