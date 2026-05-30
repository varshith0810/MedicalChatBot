# Deploy MedicalChatBot on Render

This guide deploys the Flask chat app on Render and uses a managed Qdrant endpoint for vectors. Render does not run `docker-compose.yml` for a Python web service, so use Qdrant Cloud or another internet-reachable Qdrant instance for production.

## 0. What you need before starting

- A GitHub or GitLab repository containing this project.
- A Qdrant Cloud cluster or another public HTTPS Qdrant URL.
- The Qdrant API key if your Qdrant endpoint requires authentication.
- A NVIDIA API key for the configured `writer/palmyra-med-70b-32k` model, or update `LLM_BASE_URL` and `LLM_MODEL` for your provider.
- PDFs available locally for ingestion.

## 1. Prepare the repository

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Put PDFs in `data/books/`.

   If your current PDFs are in `data/`, either move them into `data/books/` or set `DATA_DIR=data`.

3. Install dependencies locally:

   ```bash
   pip install -r requirements.txt
   ```

## 2. Create or choose a Qdrant endpoint

1. Create a Qdrant Cloud cluster, or host Qdrant where Render can reach it.
2. Copy the HTTPS cluster URL into `.env` as `QDRANT_URL`.
3. Copy the API key into `.env` as `QDRANT_API_KEY` if authentication is enabled.
4. Keep `QDRANT_COLLECTION=medical_books` unless you intentionally want a different collection name.

Example:

```ini
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=medical_books
```

## 3. Ingest your books before deploying

Run ingestion from your local machine against the same Qdrant endpoint Render will use:

```bash
python scripts/ingest_books.py
```

Confirm the script prints stored chunks/vectors. The Render web app expects the collection to already exist and contain vectors.

## 4. Push to GitHub/GitLab

```bash
git add .
git commit -m "Prepare Render deployment"
git push origin your-branch
```

If you are already working from a PR branch, push that branch and connect it in Render.

## 5. Deploy with the included Render Blueprint

1. Open the Render Dashboard.
2. Select **New +**.
3. Select **Blueprint**.
4. Connect the GitHub/GitLab repository.
5. Render detects the root-level `render.yaml` file.
6. Enter the prompted secret values:
   - `QDRANT_URL`
   - `QDRANT_API_KEY` (leave blank only if your Qdrant endpoint does not require it)
   - `NVIDIA_API_KEY`
7. Create/apply the Blueprint.

The Blueprint uses:

```bash
pip install -r requirements.txt
```

for build, and:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
```

for start.

## 6. Deploy manually instead of Blueprint

If you do not want to use `render.yaml`, create a **Web Service** manually with these values:

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120` |
| Health Check Path | `/healthz` |

Add these environment variables in the Render service settings:

| Key | Recommended value |
| --- | --- |
| `PYTHON_VERSION` | `3.10.13` |
| `DATA_DIR` | `data/books` |
| `SQLITE_DB_PATH` | `/opt/render/project/src/data/medical_chatbot.sqlite3` |
| `QDRANT_URL` | Your Qdrant Cloud URL |
| `QDRANT_API_KEY` | Your Qdrant API key |
| `QDRANT_COLLECTION` | `medical_books` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` |
| `EMBEDDING_DIM` | `384` |
| `CHUNK_SIZE` | `500` |
| `CHUNK_OVERLAP` | `50` |
| `RETRIEVAL_K` | `3` |
| `LLM_BASE_URL` | `https://integrate.api.nvidia.com/v1` |
| `LLM_MODEL` | `writer/palmyra-med-70b-32k` |
| `NVIDIA_API_KEY` | Your NVIDIA API key |

## 7. Verify the deployed app

1. Wait until the Render deploy is live.
2. Open:

   ```text
   https://your-service-name.onrender.com/healthz
   ```

   You should see JSON similar to:

   ```json
   {"service":"medical-chatbot","status":"ok"}
   ```

3. Open the root app URL:

   ```text
   https://your-service-name.onrender.com/
   ```

4. Ask a question related to the ingested PDFs.

## 8. Troubleshooting checklist

- **Build fails during dependency install:** check Render build logs for the exact package error and verify `PYTHON_VERSION=3.10.13`.
- **Deploy fails health check:** make sure the start command binds to `$PORT` and `/healthz` returns `2xx`.
- **Chat returns knowledge-base connection error:** verify `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION` match the endpoint used during ingestion.
- **No relevant answers:** rerun `python scripts/ingest_books.py` and confirm vectors were stored in the same Qdrant collection used by Render.
- **Slow first answer:** the app lazily loads the embedding model and retrieval chain on first chat request. This is expected, especially on free instances.
- **Free instance sleeps:** Render free web services may cold-start after inactivity. Use a paid instance for better latency.

## 9. Redeploy after changes

After committing and pushing changes, Render auto-deploys the connected branch by default. If auto-deploy is disabled, open the service in Render and select **Manual Deploy**.
