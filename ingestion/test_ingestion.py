import os
import shutil
import boto3
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

AWS_BUCKET = os.getenv("AWS_BUCKET_NAME")
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "../vector_store")
PDF_KEY = "test/test-file.pdf"  # Cambia esto por otro archivo si lo deseas

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedding_function = OpenAIEmbeddings()
s3 = boto3.client("s3")

def clean_vector_store():
    if os.path.exists(VECTOR_DIR):
        print(f"🧹 Eliminando base vectorial anterior en '{VECTOR_DIR}'...")
        shutil.rmtree(VECTOR_DIR)

def process_test_pdf(key):
    try:
        print("📥 Descargando PDF desde S3...")
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
        data = obj["Body"].read()
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

        preview_limit = 3
        for i, chunk in enumerate(chunks[:preview_limit]):
            print(f"📄 Fragmento {i + 1}:\n{'-'*40}\n{chunk[:500]}\n{'-'*40}\n")

        docs = [Document(page_content=chunk, metadata={"source": key}) for chunk in chunks]

        clean_vector_store()  # 🧼 Limpia antes de guardar

        print("💾 Guardando en nueva base ChromaDB...")
        Chroma.from_documents(docs, embedding_function, persist_directory=VECTOR_DIR).persist()

        print("\n✅ Proceso de prueba finalizado exitosamente.\n")
    except Exception as e:
        print(f"❌ Error procesando el archivo de prueba: {e}")

if __name__ == "__main__":
    process_test_pdf(PDF_KEY)
