# Local data directory

Umieszczaj lokalne PDF-y źródłowe w katalogu `data/` (np. `data/Przewodnik_po_historii_Polski.pdf`).

Pliki PDF, indeks Chroma (`data/chroma/`) i baza SQLite (`data/chat.db`) nie są częścią repozytorium i nie powinny być commitowane.

Po dodaniu lub zmianie PDF uruchom ingest:

```bash
uv run python -m app.ingest data/
```
