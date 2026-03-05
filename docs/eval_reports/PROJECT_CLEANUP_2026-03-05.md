# PROJECT_CLEANUP (Phase 1)

Data: 2026-03-05
Zakres: tylko Phase 1 (offline console chat + ingest + retrieval + eval), bez GUI/Phase 2 i bez nowych feature'ów produktowych.

## 1) Audyt projektu

### CLI / entrypointy
- `python -m app.ingest` -> [src/app/ingest/__main__.py](/home/bravs/projects/repos/MAG/src/app/ingest/__main__.py)
- `python -m app.chat` -> [src/app/chat/__main__.py](/home/bravs/projects/repos/MAG/src/app/chat/__main__.py)
- `python -m app.eval_chat` -> [src/app/eval_chat.py](/home/bravs/projects/repos/MAG/src/app/eval_chat.py)
- `python -m app.preflight_questions` -> [src/app/preflight_questions.py](/home/bravs/projects/repos/MAG/src/app/preflight_questions.py)
- `python -m app.bench_pdf_parsers` (narzędzie opcjonalne) -> [src/app/bench_pdf_parsers.py](/home/bravs/projects/repos/MAG/src/app/bench_pdf_parsers.py)
- Phase 2 (nietknięte): `chainlit run src/app/ui_chainlit.py` -> [src/app/ui_chainlit.py](/home/bravs/projects/repos/MAG/src/app/ui_chainlit.py)

### Moduły runtime (Phase 1)
- Orkiestracja czatu: [src/app/chat/service.py](/home/bravs/projects/repos/MAG/src/app/chat/service.py)
- Prompt/polityka nauczyciela: [src/app/dialogue/prompt_builder.py](/home/bravs/projects/repos/MAG/src/app/dialogue/prompt_builder.py), [src/app/dialogue/teacher_policy.py](/home/bravs/projects/repos/MAG/src/app/dialogue/teacher_policy.py)
- Retrieval: [src/app/retrieval/chroma_retriever.py](/home/bravs/projects/repos/MAG/src/app/retrieval/chroma_retriever.py), [src/app/retrieval/hybrid.py](/home/bravs/projects/repos/MAG/src/app/retrieval/hybrid.py), [src/app/retrieval/citations.py](/home/bravs/projects/repos/MAG/src/app/retrieval/citations.py)
- Ingest: [src/app/ingest/chroma_index.py](/home/bravs/projects/repos/MAG/src/app/ingest/chroma_index.py), [src/app/ingest/chunker.py](/home/bravs/projects/repos/MAG/src/app/ingest/chunker.py), [src/app/ingest/pdf_parsers.py](/home/bravs/projects/repos/MAG/src/app/ingest/pdf_parsers.py)
- Persistence/memory/runtime: [src/app/persistence/repositories.py](/home/bravs/projects/repos/MAG/src/app/persistence/repositories.py), [src/app/memory/summarizer.py](/home/bravs/projects/repos/MAG/src/app/memory/summarizer.py), [src/app/runtime/ollama_client.py](/home/bravs/projects/repos/MAG/src/app/runtime/ollama_client.py)

### Znalezione redundancje / martwy kod
- Martwy helper `role_from_str` w `persistence/repositories.py` (brak użyć).
- Martwy typ `ConversationState` w `types.py` (brak użyć).
- Duplikacja mapowania pytań eval/preflight (`_load_questions` i `load_questions`) do wspólnego schematu.
- Duplikacja serializacji cytowań między persistence/export.
- Artefakty repo: `__pycache__`, `src/mag.egg-info`, stare raporty timestampowe w `results/`.

### Weryfikacja wymagań bezpieczeństwa/groundedness/deterministyki
- No-context gating i polityka cytowań w `ChatService` zachowane (bez luzowania zasad groundedness).
- Parametry deterministyczne eval (`seed`, `temperature`, `top_p`, `timeout`) pozostały bez zmian semantycznych.
- Pipeline cytowań pozostał audytowalny (`sources_json` + format cytowania).

## 2) Cleanup + refactor (minimalny, bez zmiany zachowania)

### Co scalono
- Wspólny loader rekordów pytań:
  - dodano `load_question_items(...)` w [src/app/eval_io.py](/home/bravs/projects/repos/MAG/src/app/eval_io.py)
  - wykorzystano go w [src/app/eval_chat.py](/home/bravs/projects/repos/MAG/src/app/eval_chat.py) i [src/app/preflight_questions.py](/home/bravs/projects/repos/MAG/src/app/preflight_questions.py)
- Wspólna serializacja cytowań:
  - `persistence/repositories.py` i `persistence/export_jsonl.py` używają `citations_to_sources_json(...)` z `retrieval/citations.py`.

### Co uproszczono
- `ChatService._retrieve`: usunięto powtórzenia ścieżki lexical fallback przez wydzielony helper `_merge_lexical_fallback_candidates(...)` (logika i rezultat bez zmiany).
- `bench_pdf_parsers`: usunięto zbędną zmienną pomiarową (`started`) bez wpływu na wynik.

### Co usunięto
- Kod:
  - `role_from_str` z `persistence/repositories.py`.
  - `ConversationState` z `types.py`.
- Artefakty:
  - `src/mag.egg-info/`
  - wszystkie `__pycache__/` w `src/` i `tests/`
  - `.pytest_cache/`, `.ruff_cache/`
  - stare pliki `results/e2e_*`, `results/preflight_*`, `results/pdf_parser_bench_*`, stare `results/versions.json`

### Dlaczego to bezpieczne
- Przed usunięciem sprawdzone użycia (`rg`), brak importów/wywołań usuwanych symboli.
- Po każdej serii zmian uruchomione `ruff` i `pytest`.
- Zachowanie runtime Phase 1 utrzymane (brak nowych feature'ów, brak zmian kontraktów CLI).

## 3) Dev tooling

- `pyproject.toml` zachowuje `pytest` i `ruff` w `[dependency-groups].dev`.
- README uzupełniony o release quality gate z czystego env i komendy runtime check.
- `.gitignore` rozszerzony o `*.egg-info/`.

## 4) Pliki zmienione

- [.gitignore](/home/bravs/projects/repos/MAG/.gitignore)
- [README.md](/home/bravs/projects/repos/MAG/README.md)
- [pyproject.toml](/home/bravs/projects/repos/MAG/pyproject.toml)
- [src/app/eval_io.py](/home/bravs/projects/repos/MAG/src/app/eval_io.py)
- [src/app/eval_chat.py](/home/bravs/projects/repos/MAG/src/app/eval_chat.py)
- [src/app/preflight_questions.py](/home/bravs/projects/repos/MAG/src/app/preflight_questions.py)
- [src/app/persistence/repositories.py](/home/bravs/projects/repos/MAG/src/app/persistence/repositories.py)
- [src/app/persistence/export_jsonl.py](/home/bravs/projects/repos/MAG/src/app/persistence/export_jsonl.py)
- [src/app/chat/service.py](/home/bravs/projects/repos/MAG/src/app/chat/service.py)
- [src/app/bench_pdf_parsers.py](/home/bravs/projects/repos/MAG/src/app/bench_pdf_parsers.py)
- [src/app/types.py](/home/bravs/projects/repos/MAG/src/app/types.py)
- [docs/eval_reports/PROJECT_CLEANUP_2026-03-05.md](/home/bravs/projects/repos/MAG/docs/eval_reports/PROJECT_CLEANUP_2026-03-05.md)

## 5) Komendy i wyniki bramki jakości

Wykonane:
1. `rm -rf .venv` -> OK
2. `uv venv .venv` -> OK (w tym środowisku wymagało `UV_CACHE_DIR=/tmp/uv-cache` z powodu sandbox cache-permissions)
3. `uv sync` -> OK (poza sandboxem, aby użyć lokalnego cache uv)
4. `uv run ruff check .` -> OK
5. `uv run pytest` -> OK (`39 passed`)
6. `uv run python -m app.ingest --help` -> OK
7. `uv run python -m app.eval_chat --questions eval/questions_pl.jsonl --seed 1234 --timeout-seconds 90 --temperature 0.0 --top-p 1.0` -> zakończone, artefakty wygenerowane (`results/e2e_*`, `results/versions.json`)
8. `uv run python -m app.chat` (smoke; brak `--help`) -> OK (start + `/exit` bez wyjątku)

Uwaga do kroku 7:
- Proces eval przeszedł technicznie (exit code 0), ale wszystkie 50 rekordów mają status `error` przez błędy połączenia do Ollama (`RemoteDisconnected`), co wynika z runtime hosta/modelu w tym środowisku.

## 6) Znane ograniczenia (max 5)

1. `OLLAMA_HOST` z `.env` (`http://172.17.176.1:11434`) zwraca `RemoteDisconnected` dla embed requestów, więc pełny eval jakościowy nie jest miarodajny mimo wygenerowanych artefaktów.
2. `uv` w sandboxie nie ma dostępu do domyślnego cache (`~/.cache/uv`), stąd potrzeba `UV_CACHE_DIR` lub uruchomień poza sandboxem.
3. W repo nadal pozostaje opcjonalny moduł Phase 2 (`ui_chainlit.py`), celowo nietknięty.
4. Raport eval (`results/e2e_report_20260305_063205.md`) odzwierciedla błędy środowiskowe, nie regresję logiki testów jednostkowych.
5. Nie uruchamiano benchmarku parserów PDF w tej iteracji cleanup (poza zakresem obowiązkowej bramki).
