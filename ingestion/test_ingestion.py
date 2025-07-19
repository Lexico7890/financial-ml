import os
import shutil
import boto3
from botocore.exceptions import ClientError
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import botocore.exceptions

load_dotenv()

AWS_BUCKET = os.getenv("BUCKET_TEST_NAME")
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "../vector-store")
PDF_KEY = "Oscar Casas Resume.pdf"  # Cambia esto por otro archivo si lo deseas

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
s3 = boto3.client("s3")
print("BUCKET_TEST_NAME desde .env:", AWS_BUCKET)

def clean_vector_store():
    print("🧹 Eliminando colección anterior de la base vectorial...")
    try:
        if os.path.exists(VECTOR_DIR):
            shutil.rmtree(VECTOR_DIR)
            print("✅ Directorio de vector store eliminado exitosamente.")
        else:
            print("ℹ️ No existe directorio anterior para eliminar.")
    except Exception as e:
        print(f"⚠️ Error al limpiar vector store: {e}")
        # Crear el directorio si no existe
        os.makedirs(VECTOR_DIR, exist_ok=True)

def process_test_pdf(key):
    try:
        print("📥 Descargando PDF desde S3...")
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
        with obj["Body"] as body_stream:
            data = body_stream.read()
            print(f"Tipo de 'data': {type(data)}")
            print(f"Primeros 100 bytes: {data[:100]}")
            if not isinstance(data, (bytes, bytearray)):
                raise ValueError("El contenido descargado no es de tipo bytes.")
            pdf = fitz.open(stream=data, filetype="pdf")

        full_text = ""
        for page in pdf:
            full_text += page.get_text()

        if not full_text.strip():
            print("⚠️ El documento está vacío.")
            return

        print("✂️ Dividiendo el texto en fragmentos...")
        chunks = text_splitter.split_text(full_text)
        print(f"\n✅ Fragmentos generados: {len(chunks)}\n")

        for i, chunk in enumerate(chunks[:3]):
            print(f"📄 Fragmento {i + 1}:\n{'-'*40}\n{chunk[:500]}\n{'-'*40}\n")

        docs = [Document(page_content=chunk, metadata={"source": key}) for chunk in chunks]

        clean_vector_store()

        print("💾 Guardando en nueva base ChromaDB...")
        db = Chroma.from_documents(docs, embedding=embedding, persist_directory=VECTOR_DIR)
        # La persistencia es automática cuando se especifica persist_directory
        # No es necesario llamar db.persist()

        print("\n✅ Proceso finalizado exitosamente.\n")

    except ClientError as e:
        print("❌ Error al acceder a S3:")
        print("Código de error:", e.response['Error']['Code'])
        print("Mensaje:", e.response['Error']['Message'])
        print("Request ID:", e.response['ResponseMetadata']['RequestId'])
        print("Host ID:", e.response['ResponseMetadata']['HostId'])
        raise

    except Exception as e:
        print("❌ Otro error ocurrió:", str(e))
        raise

if __name__ == "__main__":
    process_test_pdf(PDF_KEY)