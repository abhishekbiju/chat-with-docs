# Chat with Docs

Locally hosted **retrieval-augmented generation (RAG)** over your own files. Put PDFs and plain text or Markdown in a folder, build a vector index, then ask questions through a **web UI** or **REST API**. Embeddings and the vector database run on your machine; answers are generated with **Ollama** so prompts and document text stay under your control.

Stack: **FastAPI**, **LangChain**, **Chroma**, **sentence-transformers** (Hugging Face), **Ollama**.

---

## Features

| Capability | Description |
|------------|-------------|
| **Web chat UI** | Open `http://localhost:8080/` for streaming answers, optional MMR retrieval, and expandable source excerpts with file name and page when available. |
| **REST API** | `POST /chat` (JSON) and `POST /chat/stream` (Server-Sent Events) for automation and custom clients. |
| **Source citations** | Each answer returns structured `sources` (excerpt, `source_file`, `page`, optional relevance score from the vector store). |
| **Conversation context** | Send `history` as prior `{ "role", "content" }` turns so follow-up questions resolve pronouns and topics. |
| **MMR retrieval** | Optional **Maximal Marginal Relevance** for more diverse context chunks when documents repeat similar wording. |
| **Multi-format ingest** | `.pdf`, `.txt`, `.md` (recursive scan under `documents/`). |
| **Robust ingest** | Per-file error handling, logging, and **safe rebuild order** (embedding model loads before wiping the old index). |
| **Health and readiness** | `GET /health`, `GET /ready` (index size + Ollama reachability), `GET /api/info` (models and paths). |
| **Configuration** | Environment variables and optional `.env` via **pydantic-settings** (Ollama URL, models, timeouts, paths). |
| **Docker** | Compose file mounts the vector store and read-only documents; `host.docker.internal` is used to reach Ollama on the host. |

---

## Quick start (local Python)

1. **Clone and enter the repo**

   ```bash
   git clone <repository-url>
   cd chat-with-docs
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   The first run downloads the embedding model from Hugging Face; ensure outbound HTTPS is allowed (set `HF_TOKEN` if you hit rate limits).

3. **Add documents** under `documents/` (PDF, `.txt`, or `.md`).

4. **Run Ollama** (separate terminal), pull a chat model, and keep the server running:

   ```bash
   ollama pull mistral
   ollama serve
   ```

5. **Build the index**

   ```bash
   python -m app.ingest
   ```

6. **Start the API**

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
   ```

7. **Use it**

   - Browser: [http://localhost:8080/](http://localhost:8080/)
   - OpenAPI: [http://localhost:8080/docs](http://localhost:8080/docs)

---

## Docker Compose

Ollama should run on the **host** (not inside this compose stack) so GPUs and model files stay where you already use them.

```bash
docker compose up --build
```

Default environment in `docker-compose.yaml` sets `OLLAMA_BASE_URL=http://host.docker.internal:11434`. On Linux, `extra_hosts: host.docker.internal:host-gateway` is included so the API container can reach the host.

After changing files in `documents/`, run ingest **on the host** (recommended) or inside a one-off container that mounts `documents` read-write:

```bash
python -m app.ingest
```

---

## Configuration (environment / `.env`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server (no trailing slash). |
| `LLM_MODEL` | `mistral` | Model name passed to Ollama. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model id for embeddings. |
| `CHROMA_PERSIST_DIR` | `app/chroma_db` | Resolved path to the Chroma persistence directory. |
| `DOCUMENTS_DIR` | `documents` | Root folder scanned by `app.ingest`. |
| `HTTPX_TIMEOUT_SECONDS` | `120` | Timeout for Ollama HTTP calls. |

Paths can be absolute or relative; they are expanded and resolved at startup.

---

## API notes

### `POST /chat`

Request body (JSON):

```json
{
  "query": "What does section 2 say about refunds?",
  "history": [
    { "role": "user", "content": "Who is the vendor?" },
    { "role": "assistant", "content": "ACME Corp." }
  ],
  "k_retrieval": 4,
  "use_mmr": false
}
```

Response: `{ "answer": "...", "sources": [ { "excerpt", "source_file", "page", "relevance_score" } ] }`. The `relevance_score` field is the raw **Chroma vector distance** (smaller = closer match); it is omitted when you use MMR-only retrieval.

### `POST /chat/stream`

Same JSON body. Response is **SSE**: `data:` lines with JSON `{"token":"..."}` while streaming, then an `event: done` with the final answer and `sources` array.

### Breaking change from v1

Earlier versions returned raw `context` strings. The API now returns **`sources`** with excerpts and metadata instead. Update any scripts that depended on `context`.

---

## Troubleshooting

- **`docker compose`: “Cannot connect to the Docker daemon … docker.sock”** — The Docker engine is not running (or not finished starting). On macOS, open **Docker Desktop** and wait until it is healthy, then run `docker info` to confirm. This message is not from the app; Compose needs a running daemon before any `Dockerfile` is used.
- **`503` on chat, “vector index is empty”** — Run `python -m app.ingest` after adding files under `documents/`.
- **`502` / connection errors** — Confirm `ollama serve` is running and `OLLAMA_BASE_URL` matches how the API reaches it (Docker → host gateway).
- **First ingest is slow** — Downloads `sentence-transformers` weights; use a stable network or a pre-populated `HF_HOME` cache.
- **Poor answers** — Increase `k_retrieval`, try `use_mmr: true`, or use a stronger Ollama model.

---

## Project layout

```
app/
  main.py          # FastAPI app, routes, static UI
  rag_logic.py     # Retrieval, Ollama chat + stream, RAGService
  schemas.py       # Request/response models
  config.py        # Settings
  ingest.py        # CLI: documents → Chroma
  static/
    index.html     # Chat UI
documents/         # Your files to index
```
