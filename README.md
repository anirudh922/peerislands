# Peerislands Repo RAG API

A simple FastAPI application built for Python 3.10.10. It analyzes a public Git
repository with a RAG pipeline using LangChain, Groq, HuggingFace embeddings, and
FAISS.

## Setup

Create and activate a virtual environment:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set your Groq API key:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

Optional model override:

```powershell
$env:GROQ_MODEL="llama-3.1-8b-instant"
```

Optional logging level:

```powershell
$env:LOG_LEVEL="INFO"
```

## Run

Start the API server:

```powershell
uvicorn app.main:app
```

Open the API in your browser:

- App: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json
- Health check: http://127.0.0.1:8000/health
- Repo analysis: http://127.0.0.1:8000/analysis?repo_url=https://github.com/tiangolo/fastapi

You can use `uvicorn app.main:app --reload` during development. If you see
`asyncio.exceptions.CancelledError` after stopping or reloading the server, it is
usually caused by the reload process cancelling an async task and can be avoided
by running without `--reload`.

## Endpoints

### `POST /embeddings/update`

Clones a public Git repository, reads source files, chunks them, creates
embeddings, and stores the FAISS index in memory. Call this once for a repo, and
call it again only when you want to refresh the index after repo changes.

The indexing step also extracts static insights such as file counts, method
signatures, and simple complexity signals.

Example:

```powershell
Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"repo_url":"https://github.com/tiangolo/fastapi"}' `
  "http://127.0.0.1:8000/embeddings/update"
```

### `GET /analysis`

Uses the existing FAISS index for the repository, retrieves relevant chunks, and
asks Groq for structured repository analysis. This endpoint does not re-clone,
re-chunk, or re-embed the repo. If the repo has not been indexed yet, it returns
a `404` asking you to call `POST /embeddings/update` first.

The response includes:

- `analysis.overview`: high-level project purpose and functionality
- `analysis.key_methods`: key method signatures with descriptions
- `analysis.complexity`: complexity level and explanation
- `analysis.noteworthy_aspects`: other important implementation details
- `static_insights`: extracted file counts, methods, and complexity signals
- `retrieved_files`: files used as RAG context

Required query parameter:

- `repo_url`: public Git repository URL

Example:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/analysis?repo_url=https://github.com/tiangolo/fastapi"
```
