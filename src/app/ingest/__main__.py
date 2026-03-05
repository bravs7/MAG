"""CLI entrypoint for PDF ingestion."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from app.config import AppConfig
from app.ingest.chroma_index import ChromaIndexer
from app.ingest.chunker import ChunkingConfig, chunk_pages
from app.ingest.pdf_parsers import load_pdf_pages_for_ingest, resolve_ingest_parser
from app.logging import configure_logging, get_logger
from app.runtime.ollama_client import OllamaClient

logger = get_logger(__name__)


def ingest(data_dir: Path, *, rebuild: bool = True) -> int:
    cfg = AppConfig.from_env()
    cfg.ensure_dirs()
    requested_parser = os.getenv("PDF_PARSER", "pymupdf")
    parser_name, fallback_note = resolve_ingest_parser(requested_parser)
    if fallback_note:
        logger.warning(fallback_note)
    logger.info("Using PDF parser for ingestion: %s", parser_name)

    chunk_cfg = ChunkingConfig(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    ollama = OllamaClient(host=cfg.ollama_host)
    indexer = ChromaIndexer(persist_dir=str(cfg.chroma_dir), collection_name=cfg.collection_name)

    if rebuild:
        indexer.rebuild_collection()

    pdf_files = sorted(data_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found under %s", data_dir)
        return 0

    total_chunks = 0
    for pdf_path in pdf_files:
        try:
            pages = load_pdf_pages_for_ingest(pdf_path, parser_name=parser_name)
        except Exception as exc:
            logger.warning(
                "Parser %s failed for %s (%s); fallback to pymupdf",
                parser_name,
                pdf_path,
                exc,
            )
            pages = load_pdf_pages_for_ingest(pdf_path, parser_name="pymupdf")
        chunks = chunk_pages(source_file=pdf_path.name, pages=pages, config=chunk_cfg)
        if not chunks:
            logger.warning("No chunks generated for %s", pdf_path)
            continue

        texts = [chunk.text for chunk in chunks]
        embeddings = ollama.embed_texts(model=cfg.embed_model, texts=texts)
        added = indexer.add_chunks(records=chunks, embeddings=embeddings)
        total_chunks += added
        logger.info("Indexed %s chunks from %s", added, pdf_path.name)

    return total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into local Chroma index")
    parser.add_argument("data_dir", nargs="?", default="data", help="Directory containing PDFs")
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Do not rebuild collection before indexing",
    )
    args = parser.parse_args()

    configure_logging(logging.INFO)
    count = ingest(Path(args.data_dir), rebuild=not args.no_rebuild)
    logger.info("Ingestion complete. Total chunks indexed: %s", count)


if __name__ == "__main__":
    main()
