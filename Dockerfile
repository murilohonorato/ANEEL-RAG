FROM python:3.11-slim

WORKDIR /app

# Dependências de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir \
    streamlit>=1.32.0 \
    qdrant-client>=1.9.0 \
    "FlagEmbedding>=1.2.0" \
    openai>=1.0.0 \
    tiktoken>=0.6.0 \
    pandas>=2.2.0 \
    pyarrow>=15.0.0 \
    python-dotenv>=1.0.0 \
    torch --index-url https://download.pytorch.org/whl/cpu \
    tqdm>=4.66.0 \
    loguru>=0.7.0

# Copiar código fonte
COPY src/ ./src/
COPY .env.example ./.env.example

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
