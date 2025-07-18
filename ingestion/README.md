# 📥 Ingestión de documentos financieros desde AWS S3

Este módulo procesa documentos PDF almacenados en un bucket S3 y construye una base vectorial persistente usando ChromaDB y OpenAI Embeddings. Los datos procesados se usarán para realizar consultas inteligentes mediante un chatbot financiero.

---

## 🚀 ¿Qué hace este script?

1. Conecta al bucket de AWS S3
2. Descarga PDFs directamente en memoria
3. Extrae el texto de los documentos usando PyMuPDF
4. Divide el texto en fragmentos (chunks)
5. Genera embeddings con OpenAI
6. Guarda todo en ChromaDB (almacenado en `ml/vector_store/`)

---

## ⚙️ Requisitos

- Claves válidas de **AWS**
- Clave API de **OpenAI**
- Archivos PDF ya cargados en un bucket S3
- Docker y docker-compose instalados

---

## 🧪 Cómo ejecutar el proceso de ingestión

1. Asegúrate de tener un archivo `.env` en la raíz del proyecto con las siguientes claves:

```env
OPENAI_API_KEY=sk-...
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

## ▶️ Ejecuta el siguiente comando desde la raíz del proyecto (financial-chat/):

```bash
docker-compose --env-file .env run --rm ingestion
```

