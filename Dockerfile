FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/models \
    PATH="/app/.venv/bin:$PATH"

RUN pip install --no-cache-dir uv==0.11.7
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Bake CPU models into the image so there is no cold-start download
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
from flashrank import Ranker; \
TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/models'); \
SparseTextEmbedding('Qdrant/bm25', cache_dir='/models'); \
Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/models')"

RUN useradd -m -u 1000 user && chown -R user:user /models
COPY --chown=user app ./app
USER user

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
