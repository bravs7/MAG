# Architecture

## Core flow
1. `app.ingest` reads PDFs with PyMuPDF.
2. Text is chunked with overlap and stable chunk IDs.
3. Chunks are embedded with local Ollama embedding model and indexed in Chroma.
4. `app.chat` retrieves top-k chunks for each user turn.
5. If context quality is too low, no-context gating response is returned.
6. Otherwise prompt is assembled deterministically and sent to Ollama.
7. Response + sources + config fingerprint are persisted in SQLite.

## Main boundaries
- `ingest`: parsing + indexing
- `retrieval`: search + citations
- `dialogue`: teacher behavior and prompt policy
- `memory`: short-term window + long-term summary trigger
- `persistence`: thread/message/state storage
- `runtime`: local model API
