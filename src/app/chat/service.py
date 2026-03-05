"""Main chat orchestration service."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from app.config import AppConfig
from app.dialogue.prompt_builder import build_prompt, default_system_rules
from app.dialogue.teacher_policy import build_no_context_response, build_teacher_rules
from app.logging import get_logger
from app.memory.summarizer import SummaryPolicy, ThreadSummarizer
from app.memory.window import estimate_tokens, take_recent_turn_messages, trim_to_token_budget
from app.persistence.repositories import Persistence
from app.retrieval.chroma_retriever import ChromaRetriever
from app.retrieval.citations import format_citations_block, from_retrieved
from app.retrieval.hybrid import (
    chunk_contains_keyword,
    chunk_keyword_hit_count,
    chunk_phrase_hit,
    ensure_top_k_contains_evidence,
    has_query_evidence,
    has_required_evidence,
    normalize_for_match,
    rerank_chunks,
    should_use_lexical_fallback,
)
from app.runtime.ollama_client import OllamaClient
from app.types import AssistantReply, ChatMessage, RetrievedChunk, SourceCitation

logger = get_logger(__name__)
MAX_CONTEXT_CHUNKS = 3
PROPER_NOUN_PATTERN = re.compile(r"\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,}\b")
PROPER_NOUN_ALLOWLIST = {
    "czy",
    "nie",
    "na",
    "w",
    "z",
    "polska",
    "polski",
    "polsce",
}


@dataclass(slots=True)
class RetrievalBundle:
    final_chunks: list[RetrievedChunk]
    candidates: list[RetrievedChunk]
    keywords: list[str]
    main_keyword: str | None
    phrase_norm: str | None
    query_evidence: bool
    lexical_fallback_used: bool


class ChatService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.ensure_dirs()

        self.persistence = Persistence(db_path=config.db_path)
        self.ollama = OllamaClient(host=config.ollama_host)
        self.retriever = ChromaRetriever(
            persist_dir=str(config.chroma_dir),
            collection_name=config.collection_name,
        )
        self.summarizer = ThreadSummarizer(self.ollama, model_name=config.model_name)

    def create_thread(self, title: str | None = None) -> str:
        thread_id = uuid.uuid4().hex
        self.persistence.threads.upsert_thread(thread_id, title=title)
        return thread_id

    def list_threads(self) -> list[dict[str, str | None]]:
        return self.persistence.threads.list_threads()

    def respond(
        self,
        thread_id: str,
        user_text: str,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        request_timeout_seconds: float | None = None,
        disable_summarization: bool = False,
    ) -> AssistantReply:
        decode_temperature = self.config.temperature if temperature is None else temperature
        decode_top_p = self.config.top_p if top_p is None else top_p

        self.persistence.threads.upsert_thread(thread_id, title=user_text[:80])
        self.persistence.messages.add_user_message(thread_id, user_text)

        state = self.persistence.thread_state.get_state(thread_id)
        all_messages = self.persistence.messages.list_messages(thread_id)

        # Exclude current user input from recent history section.
        history_before_current = all_messages[:-1]

        summary = state.summary
        recent_for_prompt = take_recent_turn_messages(history_before_current, self.config.n_turns)

        summary_policy = SummaryPolicy(
            summary_trigger_tokens=self.config.summary_trigger_tokens,
            summary_trigger_turns=self.config.summary_trigger_turns,
            keep_last_turns=self.config.keep_last_turns,
        )
        if (not disable_summarization) and self.summarizer.should_summarize(
            summary=summary,
            messages=history_before_current,
            policy=summary_policy,
        ):
            older, kept = self.summarizer.split_for_summary(
                history_before_current,
                keep_last_turns=self.config.keep_last_turns,
            )
            summary = self.summarizer.build_or_update_summary(
                existing_summary=summary,
                older_messages=older,
            )
            self.persistence.thread_state.upsert_state(
                thread_id=thread_id,
                summary=summary,
                memory_version=state.memory_version + 1,
            )
            recent_for_prompt = take_recent_turn_messages(kept, self.config.n_turns)

        summary, recent_for_prompt = trim_to_token_budget(
            summary=summary,
            recent_messages=recent_for_prompt,
            max_prompt_tokens=self.config.max_prompt_tokens,
        )

        retrieval = self._retrieve(
            user_text=user_text,
            request_timeout_seconds=request_timeout_seconds,
        )
        retrieved_chunks = retrieval.final_chunks
        retrieved_count = len(retrieved_chunks)
        best_score = max((chunk.score for chunk in retrieved_chunks), default=0.0)
        has_context = retrieved_count > 0 and best_score >= self.config.similarity_threshold
        if has_context and not retrieval.query_evidence:
            has_context = False

        if self.config.debug_retrieval:
            self._log_retrieval_debug(
                query=user_text,
                candidates=retrieval.candidates,
                main_keyword=retrieval.main_keyword,
                phrase_norm=retrieval.phrase_norm,
            )

        logger.info(
            (
                "Retrieval diagnostics: retrieved_count=%s "
                "best_score=%.4f threshold=%.4f has_context=%s"
            ),
            retrieved_count,
            best_score,
            self.config.similarity_threshold,
            has_context,
        )

        if not has_context:
            content = build_no_context_response()
            citations = []
            numeric_guard_triggered = False
        else:
            context_chunks = _select_context_chunks(
                retrieved_chunks=retrieved_chunks,
                phrase_norm=retrieval.phrase_norm,
                main_keyword=retrieval.main_keyword,
                keywords=retrieval.keywords,
            )[:MAX_CONTEXT_CHUNKS]
            if not context_chunks:
                has_context = False
                content = build_no_context_response()
                citations = []
                numeric_guard_triggered = False
            else:
                teacher_rules = build_teacher_rules(has_context=True, user_text=user_text)
                prompt = build_prompt(
                    system_rules=default_system_rules(),
                    teacher_rules=teacher_rules,
                    summary=summary,
                    recent_messages=recent_for_prompt,
                    retrieved_chunks=context_chunks,
                    user_message=user_text,
                )
                generate_kwargs: dict[str, Any] = {
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "temperature": decode_temperature,
                    "top_p": decode_top_p,
                }
                if seed is not None:
                    generate_kwargs["seed"] = seed
                if request_timeout_seconds is not None:
                    generate_kwargs["timeout_seconds"] = request_timeout_seconds
                generated = self.ollama.generate(
                    **generate_kwargs,
                ).strip()
                generated = _sanitize_generated_output(generated)
                if not generated:
                    generated = build_no_context_response()
                if _has_unsupported_numeric_claims(generated, context_chunks):
                    logger.info(
                        (
                            "Grounding guard activated: unsupported numeric claim "
                            "detected for thread %s"
                        ),
                        thread_id,
                    )
                    has_context = False
                    numeric_guard_triggered = True
                    citations = []
                    content = build_no_context_response()
                elif _looks_like_no_context_response(generated):
                    logger.info(
                        "Grounding guard activated: no-context generation despite evidence for "
                        "thread %s",
                        thread_id,
                    )
                    numeric_guard_triggered = False
                    content, citations = _build_extractive_fallback(
                        context_chunks=context_chunks,
                        phrase_norm=retrieval.phrase_norm,
                        main_keyword=retrieval.main_keyword,
                        keywords=retrieval.keywords,
                    )
                elif _has_unsupported_proper_nouns(generated, context_chunks):
                    logger.info(
                        "Grounding guard activated: unsupported proper noun detected for thread %s",
                        thread_id,
                    )
                    numeric_guard_triggered = False
                    content, citations = _build_extractive_fallback(
                        context_chunks=context_chunks,
                        phrase_norm=retrieval.phrase_norm,
                        main_keyword=retrieval.main_keyword,
                        keywords=retrieval.keywords,
                    )
                elif _should_attach_citations(generated):
                    numeric_guard_triggered = False
                    citations = from_retrieved(context_chunks)
                    content = generated + format_citations_block(citations)
                else:
                    numeric_guard_triggered = False
                    if _looks_like_no_context_response(generated):
                        has_context = False
                        content = build_no_context_response()
                    else:
                        content = generated
                    citations = []

        token_count = estimate_tokens(content)
        fingerprint = self.config.fingerprint()
        fingerprint["llm"]["temperature"] = decode_temperature
        fingerprint["llm"]["top_p"] = decode_top_p
        if seed is not None:
            fingerprint["llm"]["seed"] = seed
        fingerprint["retrieval"] = [
            {"chunk_id": chunk.chunk_id, "score": chunk.score, "source_file": chunk.source_file}
            for chunk in retrieved_chunks
        ]
        fingerprint["retrieval_summary"] = {
            "retrieved_count": retrieved_count,
            "best_score": best_score,
            "threshold": self.config.similarity_threshold,
            "has_context": has_context,
            "candidate_count": len(retrieval.candidates),
            "main_keyword": retrieval.main_keyword,
            "phrase_norm": retrieval.phrase_norm,
            "query_evidence": retrieval.query_evidence,
            "lexical_fallback_used": retrieval.lexical_fallback_used,
            "numeric_guard_triggered": numeric_guard_triggered,
        }

        self.persistence.messages.add_assistant_message(
            thread_id=thread_id,
            content=content,
            model=self.config.model_name,
            token_count=token_count,
            sources=citations,
            config_fingerprint=fingerprint,
        )

        return AssistantReply(
            content=content,
            sources=citations,
            token_count=token_count,
            config_fingerprint=fingerprint,
        )

    def _retrieve(
        self,
        *,
        user_text: str,
        request_timeout_seconds: float | None = None,
    ) -> RetrievalBundle:
        embed_kwargs: dict[str, Any] = {
            "model": self.config.embed_model,
            "texts": [user_text],
        }
        if request_timeout_seconds is not None:
            embed_kwargs["timeout_seconds"] = request_timeout_seconds
        embeddings = self.ollama.embed_texts(**embed_kwargs)
        if not embeddings:
            return RetrievalBundle(
                final_chunks=[],
                candidates=[],
                keywords=[],
                main_keyword=None,
                phrase_norm=None,
                query_evidence=False,
                lexical_fallback_used=False,
            )

        embedding = embeddings[0]
        candidate_pool = max(self.config.top_k * 5, 25)
        vector_candidates = self.retriever.retrieve(
            query_embedding=embedding,
            top_k=candidate_pool,
        )
        reranked_candidates, keyword_query = rerank_chunks(vector_candidates, user_text)

        lexical_fallback_used = False
        if should_use_lexical_fallback(reranked_candidates, keyword_query):
            merged_candidates = self._merge_lexical_fallback_candidates(
                candidates=reranked_candidates,
                phrase_terms=keyword_query.phrase_terms,
                search_terms=keyword_query.lexical_terms,
                limit=candidate_pool,
            )
            if merged_candidates:
                lexical_fallback_used = True
                reranked_candidates, keyword_query = rerank_chunks(merged_candidates, user_text)

        final_chunks = ensure_top_k_contains_evidence(
            reranked_candidates,
            keyword_query,
            self.config.top_k,
        )
        if (
            not has_required_evidence(final_chunks, keyword_query)
            and not lexical_fallback_used
            and (keyword_query.phrase_terms or keyword_query.keywords)
        ):
            merged_candidates = self._merge_lexical_fallback_candidates(
                candidates=reranked_candidates,
                phrase_terms=keyword_query.phrase_terms,
                search_terms=keyword_query.lexical_terms,
                limit=candidate_pool,
            )
            if merged_candidates:
                lexical_fallback_used = True
                reranked_candidates, keyword_query = rerank_chunks(merged_candidates, user_text)
                final_chunks = ensure_top_k_contains_evidence(
                    reranked_candidates,
                    keyword_query,
                    self.config.top_k,
                )
        query_evidence = has_query_evidence(final_chunks, keyword_query)
        return RetrievalBundle(
            final_chunks=final_chunks,
            candidates=reranked_candidates,
            keywords=keyword_query.keywords,
            main_keyword=keyword_query.main_keyword,
            phrase_norm=keyword_query.phrase_norm,
            query_evidence=query_evidence,
            lexical_fallback_used=lexical_fallback_used,
        )

    def get_thread_messages(self, thread_id: str) -> list[ChatMessage]:
        return self.persistence.messages.list_messages(thread_id)

    def _lexical_fallback(
        self,
        *,
        phrase_terms: list[str],
        search_terms: list[str],
        limit: int,
    ) -> list[RetrievedChunk]:
        gathered: list[RetrievedChunk] = []
        for phrase in phrase_terms:
            gathered.extend(
                self.retriever.retrieve_by_document_contains(
                    phrase=phrase,
                    limit=max(5, min(20, limit // 2)),
                )
            )
        per_term_limit = max(5, min(15, limit // 2))
        for term in search_terms:
            gathered.extend(
                self.retriever.retrieve_by_document_contains(
                    phrase=term,
                    limit=per_term_limit,
                )
            )
        return gathered

    def _merge_lexical_fallback_candidates(
        self,
        *,
        candidates: list[RetrievedChunk],
        phrase_terms: list[str],
        search_terms: list[str],
        limit: int,
    ) -> list[RetrievedChunk]:
        lexical_fallback_candidates = self._lexical_fallback(
            phrase_terms=phrase_terms,
            search_terms=search_terms,
            limit=limit,
        )
        if not lexical_fallback_candidates:
            return []
        return _merge_chunks(candidates, lexical_fallback_candidates)

    def _log_retrieval_debug(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
        main_keyword: str | None,
        phrase_norm: str | None,
    ) -> None:
        top_n = min(len(candidates), self.config.retrieval_debug_top_n)
        logger.info(
            (
                "DEBUG_RETRIEVAL query=%r main_keyword=%s phrase_norm=%s "
                "candidates=%s top_n=%s"
            ),
            query,
            main_keyword,
            phrase_norm,
            len(candidates),
            top_n,
        )
        for idx, chunk in enumerate(candidates[:top_n], start=1):
            preview = " ".join(chunk.text.split())[:160]
            logger.info(
                (
                    "DEBUG_RETRIEVAL #%02d page=%s score=%.4f chunk_id=%s "
                    "contains_keyword=%s phrase_hit=%s keyword_hits_main=%s text=%s"
                ),
                idx,
                chunk.page,
                chunk.score,
                chunk.chunk_id,
                chunk_contains_keyword(chunk, main_keyword),
                chunk_phrase_hit(chunk, phrase_norm),
                chunk_keyword_hit_count(chunk, [main_keyword] if main_keyword else []),
                preview,
            )


def _should_attach_citations(generated_text: str) -> bool:
    return not _looks_like_no_context_response(generated_text)


def _looks_like_no_context_response(generated_text: str) -> bool:
    normalized = generated_text.lower()
    negative_markers = [
        "nie wiem na podstawie dostarczonych materiałów",
        "nie wiem na podstawie dostarczonych materialow",
        "w materiałach nie znalazłem informacji",
        "w materialach nie znalazlem informacji",
        "w dostarczonych materiałach nie ma informacji",
        "w dostarczonych materialach nie ma informacji",
        "w dostarczonym kontekście nie ma informacji",
        "w dostarczonym kontekscie nie ma informacji",
        "nie ma informacji na temat",
        "nie znalazłem informacji na temat",
        "nie znalazlem informacji na temat",
    ]
    return any(marker in normalized for marker in negative_markers)


def _has_unsupported_numeric_claims(
    generated_text: str,
    retrieved_chunks: list[RetrievedChunk],
) -> bool:
    years = set(re.findall(r"\b\d{3,4}\b", generated_text))
    if not years:
        return False

    context_text = "\n".join(chunk.text for chunk in retrieved_chunks)
    return any(year not in context_text for year in years)


def _has_unsupported_proper_nouns(
    generated_text: str,
    retrieved_chunks: list[RetrievedChunk],
) -> bool:
    if not generated_text.strip() or not retrieved_chunks:
        return False

    context_text = normalize_for_match("\n".join(chunk.text for chunk in retrieved_chunks))
    if not context_text:
        return False

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", generated_text)
        if sentence.strip()
    ]
    for sentence in sentences:
        words = re.findall(r"\b\w+\b", sentence, flags=re.UNICODE)
        first_word = normalize_for_match(words[0]) if words else None

        for noun in PROPER_NOUN_PATTERN.findall(sentence):
            noun_norm = normalize_for_match(noun)
            if not noun_norm:
                continue
            if noun_norm in PROPER_NOUN_ALLOWLIST:
                continue
            if first_word and noun_norm == first_word:
                continue
            if noun_norm not in context_text:
                return True
    return False


def _build_extractive_fallback(
    *,
    context_chunks: list[RetrievedChunk],
    phrase_norm: str | None,
    main_keyword: str | None,
    keywords: list[str],
) -> tuple[str, list[SourceCitation]]:
    evidence_chunk = _pick_evidence_chunk(
        context_chunks=context_chunks,
        phrase_norm=phrase_norm,
        main_keyword=main_keyword,
        keywords=keywords,
    )
    snippet = _extract_chunk_snippet(evidence_chunk.text, main_keyword=main_keyword)
    content = (
        "Na podstawie dostarczonego materiału mogę powiedzieć: "
        f"{snippet}\n\nCzy to wyjaśnienie jest dla Ciebie zrozumiałe?"
    )
    citations = from_retrieved([evidence_chunk])
    return content + format_citations_block(citations), citations


def _sanitize_generated_output(generated_text: str) -> str:
    cleaned = re.sub(
        r"\[(?:Źródło|Zrodlo):[^\]]+\]",
        "",
        generated_text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*(?:Źródło|Zrodlo):.*$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _merge_chunks(
    primary: list[RetrievedChunk], extra: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in primary}
    for chunk in extra:
        existing = merged.get(chunk.chunk_id)
        if existing is None or chunk.score > existing.score:
            merged[chunk.chunk_id] = chunk
    return list(merged.values())


def _select_context_chunks(
    *,
    retrieved_chunks: list[RetrievedChunk],
    phrase_norm: str | None,
    main_keyword: str | None,
    keywords: list[str],
) -> list[RetrievedChunk]:
    if phrase_norm:
        phrase_hits = [
            chunk for chunk in retrieved_chunks if chunk_phrase_hit(chunk, phrase_norm)
        ]
        if phrase_hits:
            return phrase_hits

    if main_keyword:
        keyword_hits = [
            chunk for chunk in retrieved_chunks if chunk_contains_keyword(chunk, main_keyword)
        ]
        if keyword_hits:
            return keyword_hits

    if len(keywords) >= 2:
        two_keyword_hits = [
            chunk
            for chunk in retrieved_chunks
            if chunk_keyword_hit_count(chunk, keywords) >= 2
        ]
        if two_keyword_hits:
            return two_keyword_hits

    return []


def _pick_evidence_chunk(
    *,
    context_chunks: list[RetrievedChunk],
    phrase_norm: str | None,
    main_keyword: str | None,
    keywords: list[str],
) -> RetrievedChunk:
    if phrase_norm:
        for chunk in context_chunks:
            if chunk_phrase_hit(chunk, phrase_norm):
                return chunk
    if main_keyword:
        for chunk in context_chunks:
            if chunk_contains_keyword(chunk, main_keyword):
                return chunk
    if len(keywords) >= 2:
        for chunk in context_chunks:
            if chunk_keyword_hit_count(chunk, keywords) >= 2:
                return chunk
    return context_chunks[0]


def _extract_chunk_snippet(text: str, *, main_keyword: str | None = None) -> str:
    compact = " ".join(text.split()).strip()
    if not compact:
        return "w materiale jest krótka wzmianka na ten temat."

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", compact) if part.strip()]
    selected_sentences = sentences
    if main_keyword and sentences:
        main_keyword_norm = normalize_for_match(main_keyword)
        keyword_sentences = [
            sentence
            for sentence in sentences
            if main_keyword_norm in normalize_for_match(sentence)
        ]
        if keyword_sentences:
            selected_sentences = keyword_sentences

    snippet = (
        " ".join(selected_sentences[:2]) if selected_sentences else compact[:260].rstrip()
    )
    if len(snippet) > 320:
        snippet = snippet[:317].rstrip() + "..."
    if snippet and snippet[-1] not in ".!?":
        snippet += "."
    return snippet
