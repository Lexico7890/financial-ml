import os
import boto3
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.document_loaders import PyMuPDFLoader
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

AWS_BUCKET = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "../vector_store")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedding_function = OpenAIEmbeddings()

s3 = boto3.client("s3")

# Estadísticas
total_pdfs = 0
total_chunks = 0
errors = []

def get_pdf_keys(bucket):
    response = s3.list_objects_v2(Bucket=bucket)
    if "Contents" not in response:
        return []
    return [item["Key"] for item in response["Contents"] if item["Key"].endswith(".pdf")]

def process_pdf_from_s3(key):
    global total_chunks
    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
        data = obj["Body"].read()
        pdf = fitz.open(stream=data, filetype="pdf")

        full_text = ""
        for page in pdf:
            full_text += page.get_text()

        if not full_text.strip():
            raise ValueError("Documento vacío")

        # División en chunks
        chunks = text_splitter.split_text(full_text)
        total_chunks += len(chunks)

        # Construcción de documentos
        docs = [Document(page_content=chunk, metadata={"source": key}) for chunk in chunks]
        return docs
    except Exception as e:
        errors.append((key, str(e)))
        return []

def main():
    global total_pdfs
    print("🚀 Iniciando proceso de ingestión desde S3...")

    pdf_keys = get_pdf_keys(AWS_BUCKET)
    all_docs = []

    for key in pdf_keys:
        print(f"📄 Procesando: {key}")
        docs = process_pdf_from_s3(key)
        if docs:
            total_pdfs += 1
            all_docs.extend(docs)

    if not all_docs:
        print("⚠️ No se procesaron documentos válidos. Abortando.")
        return

    print("💾 Guardando documentos en la base vectorial (ChromaDB)...")
    Chroma.from_documents(all_docs, embedding_function, persist_directory=VECTOR_DIR).persist()

    print("\n✅ Proceso finalizado.")
    print("📊 Resumen:")
    print(f"  - Archivos PDF procesados: {total_pdfs}")
    print(f"  - Fragmentos generados:    {total_chunks}")
    print(f"  - Errores:                 {len(errors)}")

    if errors:
        print("\n❌ Errores encontrados:")
        for key, err in errors:
            print(f"  - {key}: {err}")

if __name__ == "__main__":
    main()
