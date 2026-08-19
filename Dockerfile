FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the embedding + reranker ONNX models into the image so the first
# request is fast and the container doesn't depend on model CDNs at runtime.
RUN python -c "from langchain_community.embeddings import FastEmbedEmbeddings; FastEmbedEmbeddings(); from langchain_classic.retrievers.document_compressors import FlashrankRerank; FlashrankRerank(model='ms-marco-MiniLM-L-12-v2')"

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]