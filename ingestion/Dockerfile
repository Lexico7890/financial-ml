FROM python:3.10-slim

WORKDIR /app

COPY ../requirements.txt ./requirements.txt
COPY ingestion/ ./ingestion/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "ingestion/process_s3_documents.py"]
