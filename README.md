# MedicalChatBot

MedicalChatBot is a Flask-based retrieval-augmented generation (RAG) assistant for medical books. The current build uses:

- **Qdrant** for vector search
- **SQLite + FTS5** for book/chunk metadata and text search
- **BAAI/bge-small-en-v1.5** for default embeddings
- **LangChain** for retrieval and answer generation
- **Render-ready deployment** with `render.yaml` and a `/healthz` endpoint

> Medical safety note: this project is an educational assistant over your ingested documents. It is not a replacement for licensed clinical care or emergency services.

## Local setup

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
pip install -r requirements.txt
cp .env.example .env

Update `.env` with your model/API settings. For the included NVIDIA-hosted model, set `NVIDIA_API_KEY`.

The default NVIDIA LLM is now `nvidia/llama-3.3-nemotron-super-49b-v1.5`, a newer Nemotron chat/reasoning model that is better suited for RAG-style medical-book Q&A than the previous Palmyra default.

## Start Qdrant locally

```bash
docker compose up -d qdrant
```

## Add books

Place PDF files in:

```text
data/books/
```

If you already have PDFs in `data/`, either move them into `data/books/` or set `DATA_DIR=data` in `.env`.

## Ingest books

```bash
python scripts/ingest_books.py
```

The pipeline loads PDFs, cleans text, chunks pages, deduplicates chunks, stores metadata in SQLite, and upserts vectors into Qdrant.
```

Update `.env` with your model/API settings. For the included NVIDIA-hosted model, set `NVIDIA_API_KEY`.

## Start Qdrant locally

```bash
docker compose up -d qdrant
```

## Add books

Place PDF files in:

```text
data/books/
```

If you already have PDFs in `data/`, either move them into `data/books/` or set `DATA_DIR=data` in `.env`.

## Ingest books

```bash
python scripts/ingest_books.py
```

The pipeline loads PDFs, cleans text, chunks pages, deduplicates chunks, stores metadata in SQLite, and upserts vectors into Qdrant.

## Run the app locally

```bash
python app.py
```
Open <http://localhost:5000>.

## Render deployment

This repository includes `render.yaml` for Render Blueprint deployment and a detailed deployment guide.

Read the full step-by-step guide here:

```text
docs/render-deployment.md
```

High-level flow:

1. Create or choose a Qdrant Cloud/public Qdrant endpoint.
2. Put PDFs in `data/books/`.
3. Set `QDRANT_URL`, `QDRANT_API_KEY`, and model credentials in `.env`.
4. Run `python scripts/ingest_books.py` locally against that Qdrant endpoint.
5. Push the repo to GitHub/GitLab.
6. In Render, create a Blueprint from `render.yaml` or manually create a Python Web Service.
7. Add the prompted secrets and deploy.
8. Verify `/healthz`, then ask a question in the UI.
Open <http://localhost:5000>.

## Render deployment

This repository includes `render.yaml` for Render Blueprint deployment.

1. Push the repository to GitHub/GitLab.
2. In Render, create a new Blueprint from the repository.
3. Provide the required secret values when prompted:
   - `QDRANT_URL` - use a managed Qdrant Cloud URL or another reachable Qdrant endpoint.
   - `NVIDIA_API_KEY` - used by the configured NVIDIA-compatible OpenAI endpoint.
4. Deploy the service.
Render runs:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
```

The health check endpoint is:

```text
/healthz
```

### Important deployment notes

- Render does not run this repo's `docker-compose.yml` for a Python web service, so use Qdrant Cloud or another reachable Qdrant endpoint for deployment.
- Render web services have ephemeral filesystems unless you add a disk. For production ingestion, prefer a persistent database and managed Qdrant instance.
- The app lazily initializes the embedding model and Qdrant retriever on the first chat request, so `/healthz` remains fast and reliable during deploy checks.
- Run ingestion before expecting answers from the chat UI; the Qdrant collection must already exist and contain vectors.

## Environment variables

See `.env.example` for all supported variables:

- `DATA_DIR`
- `SQLITE_DB_PATH`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIM`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVAL_K`
- `LLM_BASE_URL`
- `LLM_MODEL`
- `NVIDIA_API_KEY`
- `PORT`

## Project structure

```text
src/config.py
src/storage/db.py
src/storage/schema.sql
src/storage/repositories.py
src/rag/embeddings.py
src/rag/vector_store.py
src/ingestion/loaders.py
src/ingestion/cleaners.py
src/ingestion/chunking.py
src/ingestion/dedup.py
src/ingestion/pipeline.py
scripts/ingest_books.py
templates/chat.html
static/style.css
docs/render-deployment.md
render.yaml
docker-compose.yml
```
