FROM python:3.11-slim

WORKDIR /app

# System deps: supervisor to manage two processes (backend + frontend)
RUN apt-get update && apt-get install -y --no-install-recommends \
        supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (separate layer — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding + re-ranking models (~180 MB baked into image layer)
# This means the container starts instantly without needing internet access at runtime
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy the full project (venv, .env, .git excluded via .dockerignore)
# data/chroma_db (15 MB) and data/processed/chunks.json are included
COPY . .

# Copy supervisord config into the standard location
COPY supervisord.conf /etc/supervisor/conf.d/app.conf

# HuggingFace Spaces requires the app to listen on port 7860
EXPOSE 7860

# Launch both backend (port 8000, internal) and frontend (port 7860, public)
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/app.conf"]
