# Phase 1 Hardening - Eval Summary (2026-03-04)

## Co zmieniłem
- Ulepszono ekstrakcję `phrase_norm`/`phrase_terms` w hybrydowym retrieval:
  - preferencja dla par informacyjnych (`zjazd gnieznienski`, `boleslaw chrobry`, `swiety wojciech`, `metropolia gnieznienska`),
  - lepsze wykrywanie encji dla pytań typu `Kim był X?` / `Kto to był X?`,
  - lista kilku wariantów `phrase_terms` zamiast pojedynczej frazy.
- Utwardzono wybór `top_k`: finalny zestaw chunków musi zawierać evidence chunk (`phrase_hit` lub >=2 keyword hits), a gdy go brakuje, wymuszany jest lexical fallback.
- Dodano sanitizację tekstu modelu: usuwanie cytowań wstawionych przez LLM; cytowania są dopinane wyłącznie przez system.
- Dodano CLI:
  - `python -m app.preflight_questions` (sprawdzenie pokrycia pytań w Chroma),
  - `python -m app.generate_questions_from_pdf` (deterministyczna generacja pytań z evidence).
- Dodano testy jednostkowe dla phrase extraction, preflight i generatora.

## Pliki wynikowe
- Preflight:
  - `results/preflight_20260304_185945.csv`
  - `results/preflight_20260304_185945.md`
- Eval chat:
  - `results/e2e_chat_20260304_190000.jsonl`
  - `results/e2e_chat_20260304_190000.csv`
  - `results/e2e_report_20260304_190000.md`
  - `results/versions.json`
- Generator pytań z PDF:
  - `eval/questions_from_pdf.jsonl` (wygenerowano 64 pytania z 50 chunków)

## Najważniejsze metryki
- Preflight (`eval/questions_pl.jsonl`):
  - `total_questions`: 50
  - `supported`: 39
  - `not_supported`: 11
  - `support_rate`: 78.0%
- Powiązanie preflight -> eval:
  - `has_context=True` dla supported: `11/39` (28.2%)
  - `has_context=True` dla not_supported: `4/11` (36.4%)  # sygnał do dalszego strojenia quality-gating
- Eval (`results/e2e_report_20260304_190000.md`):
  - `ok_rate`: 100.0% (50/50), `timeout_count`: 0
  - `citation_rate_when_has_context=True`: 100.0% (15/15)
  - `no_citation_rate_when_has_context=False`: 100.0% (35/35)
  - `forbidden_marker_hits`: 0
  - `avg_latency`: 1.752s, `median_latency`: 1.903s
