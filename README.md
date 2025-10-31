# Shokti — energy insights backend

A compact backend for extracting, indexing, and serving household energy insights. The codebase contains a small FastAPI service that uses a RAG-style pipeline to search time-series power consumption records stored as JSON and CSV, builds a FAISS vector index of human-readable summaries, and uses a Groq LLM client to generate contextual responses.

This repository is an application scaffold focused on two goals:

- Convert numeric power records into human-readable documents and index them with embeddings (see `rag_pipline.py`).
- Expose a small chat endpoint that retrieves similar records from the vector store, builds a contextual prompt and forwards the prompt to an LLM via the Groq client (see `main.py` and `main1.py`).

## Quick Lookups

- Python: 3.12+
- Key libraries: FastAPI, LangChain (community extensions), FAISS, Hugging Face sentence-transformers, Groq client.
- Vector index output: `./data/vector_index/user_001_faiss` (FAISS files).

## Repository layout

- `main.py` — primary FastAPI app with `/chat` endpoint (expects JSON with `message` and optional `session_id`). Uses `rag_pipline.rag_pipeline()` to run similarity search and forwards context to Groq models.
- `rag_pipline.py` — loader/transformation script: reads `data/user_001.json`, transforms numeric records into text documents with metadata, embeds them with `sentence-transformers/all-MiniLM-L6-v2`, and builds/saves a FAISS index.
- `pyproject.toml` — dependency manifest used for installation.
- `data/` — example dataset and pre-built vector index (CSV, JSON, and `vector_index/` directory).
- `notebooks/` — exploratory notebooks; `adding.ipynb` contains experiments and notes about the forecasting model.

## Environment and prerequisites

1. Install Python 3.12 or newer.
2. Create and activate a virtual environment:

```bash
uv pip install .
```

## Required environment variables

Set the following variables before running the API servers:

- `GROQ_API_KEY` — API key for the Groq LLM client.

Create a `.env` file at the project root or set these in your environment. The code uses `python-dotenv` to load `.env` automatically.

## Building the vector index

The `rag_pipline.py` script constructs human-readable document summaries from raw JSON records and builds a FAISS vectorstore.

To generate or refresh the index:

```bash
python rag_pipline.py
```

On success, the FAISS index and auxiliary files will be saved under `./data/vector_index/user_001_faiss`.

## Running the API

Start the main API (example using `main.py`):

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints and payloads:

- `POST /chat` (from `main.py`)
	- Request JSON: `{ "message": "<user text>", "session_id": "<optional>" }`
	- Response JSON: `{ "session_id": "...", "reply": "..." }`

## Data notes

- `data/cycled_hourly_power.csv` — example CSV time-series (hourly) used by `main.py` for simple analytics.
- `data/user_001.json` — per-record JSON entries used by `rag_pipline.py` to create the document set.
- `data/vector_index/user_001_faiss` — built FAISS index (if present).

The transformation in `rag_pipline.transform_record_to_document` maps numeric columns into a readable summary and attaches metadata such as `datetime`, `power_level`, `time_period`, and `day_type`.

## Development notes

- Sessions are stored in memory (`sessions` dict) in both `main.py`.
- CORS is currently configured with `allow_origins=["*"]`. Restrict origins in deployment.
- Model rotation: a list of model names is defined in `MODELS`. The app attempts models in order and falls back on the next if a request errors.
- Error handling is minimal.

## Notebooks and exploration

`notebooks/adding.ipynb` contains experiments with the forecasting model used during development. The result of the fine tune is reflected in the main codebase.
