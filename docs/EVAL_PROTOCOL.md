# Evaluation protocol

## Baselines
- RAG ON (normal retrieval)
- RAG OFF / empty context (force no-context path)

## Per-question logs
- latency_seconds
- retrieved_count
- best_similarity
- teacher_quality_manual (0-2)
- groundedness_manual (0-2)
- clarity_manual (0-2)

## Output
Store timestamped CSV/JSONL in `results/` and include run metadata in `results/versions.json`.
