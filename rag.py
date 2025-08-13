# rag.py
import os
from langchain_ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# 1. Configuración del modelo Ollama
ollama_model = Ollama(
    base_url="http://localhost:11434",
    model="llama3.2",
    timeout=60  # en segundos
)

# 2. Cargar el README.md de rsk-cli
try:
    print("📖 Leyendo archivo README.md...")
    with open("./docs/README.md", "r", encoding="utf-8") as f:
        readme_content = f.read()
    print(f"✅ Archivo leído exitosamente, tamaño: {len(readme_content)} caracteres")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents([
        Document(page_content=readme_content)
    ])

    print(f"📄 Documentos creados: {len(docs)}")

    # 3. Crear embeddings y almacenar en ChromaDB (local)
    print("🔄 Creando embeddings con 'nomic-embed-text'...")
    vector_store = Chroma.from_documents(
        docs,
        OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434",
            timeout=120
        ),
        collection_name="rsk-cli-docs"
    )
    print("✅ Vector store creado exitosamente")

    # 4. Función para consultar
    def ask_question(question):
        try:
            print("🔍 Buscando documentos similares...")
            results = vector_store.similarity_search(question, k=3)

            context = "\n".join([r.page_content for r in results])

            prompt = f"""
Responde a la siguiente pregunta usando SOLO la información del contexto.
Si no está en el contexto, responde "No tengo esa información".

Contexto:
{context}

Pregunta: {question}
Respuesta:
"""

            print("🤖 Generando respuesta...")
            response = ollama_model.invoke(prompt)
            print("🤖 Respuesta:", response)
        except Exception as e:
            print(f"❌ Error al procesar la pregunta: {e}")

    # 5. Ejemplo de uso
    ask_question("¿Qué es rsk-cli y para qué se utiliza?")

except FileNotFoundError:
    print("❌ Error: No se encontró el archivo ./docs/README.md")
    print("💡 Asegúrate de que el archivo existe en la ruta correcta")
except Exception as e:
    print(f"❌ Error al procesar: {e}")
