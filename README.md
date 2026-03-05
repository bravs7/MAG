# MAG - Local Educational Chatbot Teacher (PoC)

Offline-first educational chatbot focused on Polish history. The app uses local PDFs, ChromaDB retrieval, Ollama generation/embeddings, short + long-term memory, and SQLite persistence.

## Requirements
- Linux/WSL Ubuntu (project on Linux filesystem, not `/mnt/c`)
- Python 3.12
- `uv` (recommended workflow)
- Ollama running locally (`OLLAMA_HOST`), with downloaded models

## Project setup (WSL)
```bash
cd /home/bravs/projects/repos/MAG
uv venv .venv
source .venv/bin/activate
uv sync
```

## Install optional PDF benchmark parsers
```bash
uv sync --extra pdfbench
```

## Models (Ollama)
```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull nomic-embed-text
```

## Environment
Copy `.env.example` to `.env` and adjust as needed:
- `DB_PATH`, `CHROMA_DIR`
- `OLLAMA_HOST`, `MODEL_NAME`, `EMBED_MODEL`
- `ANONYMIZED_TELEMETRY` (`False` recommended for offline-first Chroma usage)
- `PDF_PARSER` (`pymupdf|pdfplumber|pypdf|pdfminer`, fallback to `pymupdf`)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `SIMILARITY_THRESHOLD`
- `DEBUG_RETRIEVAL`, `RETRIEVAL_DEBUG_TOP_N`
- `N_TURNS`, `SUMMARY_TRIGGER_TOKENS`, `SUMMARY_TRIGGER_TURNS`
- `TEMPERATURE`, `TOP_P`

If Ollama runs on Windows and MAG runs in WSL, set `OLLAMA_HOST` in WSL to the Windows host IP:
```bash
WIN_HOST_IP="$(ip route | awk '/^default/ {print $3; exit}')"
export OLLAMA_HOST="http://${WIN_HOST_IP}:11434"
```
Set the same value in `.env` for persistent configuration.

## Dane (PDF)
PDF-y źródłowe trzymaj w katalogu `data/` (np. `data/Przewodnik_po_historii_Polski_PL_internet.pdf`).  
Po dodaniu lub zmianie PDF uruchom ponownie krok **1) ingest**, aby przebudować/odświeżyć indeks w ChromaDB.


## Phase 1 workflow (console only)
```bash
# 1) ingest (build/refresh vector index from PDFs)
uv run python -m app.ingest data/

# 2) chat (offline teacher console)
uv run python -m app.chat

# 3) preflight (optional support check before eval)
uv run python -m app.preflight_questions --questions eval/questions_pl.jsonl

# 4) deterministic eval gate
uv run python -m app.eval_chat --questions eval/questions_pl.jsonl --seed 1234 --timeout-seconds 90 --temperature 0.0 --top-p 1.0
```

## Quality gate (clean env)
```bash
# recreate environment from scratch
rm -rf .venv
uv venv .venv
source .venv/bin/activate
uv sync

# dev tools come from [dependency-groups].dev
uv run ruff check .
uv run pytest

# runtime sanity check
uv run python -m app.ingest --help
```

Deterministic eval gate is run in the workflow above (step 4).

`app.chat` has no `--help`; validate startup with:
```bash
uv run python -m app.chat
```

## Notes
- If retrieval returns no usable context, the assistant must not guess facts.
- The app stores per-thread state in SQLite and keeps source citations in `messages.sources_json`.
- Phase 2 / Chainlit GUI code stays in the repo, but it is out of scope for the Phase 1 target and quality gate.
