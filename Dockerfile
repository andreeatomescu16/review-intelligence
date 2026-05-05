FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck scripts and other minimal utilities
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached independently of app code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model at build time so it is baked
# into the image. This avoids a network fetch on every container start and
# makes cold-start latency deterministic.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')"

# Copy application source
COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
