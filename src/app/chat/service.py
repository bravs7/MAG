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
    "ciebie",
    "tobie",
    "twoje",
    "twoj",
    "twoja",
    "twoim",
    "polska",
    "polski",
    "polsce",
}
SUMMARY_FOLLOWUP_MARKERS = (
    "przypomnij",
    "wrocmy",
    "wroc",
    "co mowiles",
    "co mowilas",
    "o czym mowiles",
    "o czym mowilas",
    "to co mowiles",
    "to co mowilas",
)
FOLLOWUP_TOPIC_STOPWORDS = {
    "podsumuj",
    "podsumowanie",
    "stresc",
    "streszcz",
    "przypomnij",
    "wrocmy",
    "wroc",
    "najwazniejsze",
    "informacje",
    "tego",
    "mowiles",
    "mowilas",
    "zdaniach",
    "dwoch",
}
META_INTENT_MARKERS = (
    "od teraz",
    "ustaw",
    "prosze ustaw",
    "odpowiadaj krotko",
    "odpowiadaj normalnie",
    "odpowiadaj szeroko",
    "tryb normalny",
    "tryb krotki",
    "tryb rozszerzony",
    "styl odpowiedzi",
    "preferencje",
    "jakie mam",
    "jakie sa moje",
    "jaka dlugosc",
    "nie zadawaj pytania",
    "bez pytania",
    "zadawaj pytania",
    "pytanie kontrolne",
)
META_QUERY_MARKERS = (
    "jakie mam ustawione preferencje",
    "jakie mam preferencje",
    "jakie sa moje preferencje",
    "jaka dlugosc odpowiedzi",
    "jaka dlugosc mam ustawiona",
    "jaki styl odpowiedzi",
    "jaki mam styl",
    "jakie ustawienia",
)
DISABLE_CHECK_QUESTION_MARKERS = (
    "nie zadawaj pytania kontrolnego",
    "nie zadawaj pytania sprawdzajacego",
    "bez pytania kontrolnego",
    "bez pytania sprawdzajacego",
    "bez pytania",
)
ENABLE_CHECK_QUESTION_MARKERS = (
    "zadawaj pytanie kontrolne",
    "zadawaj pytania kontrolne",
    "zadawaj pytanie sprawdzajace",
    "zadawaj pytania sprawdzajace",
)
DEFAULT_CHECK_QUESTION = "Czy to jest dla Ciebie zrozumiałe?"
DEFAULT_ANSWER_STYLE = "normal"
ANSWER_STYLE_VALUES = {"normal", "short", "extended"}
STYLE_FACT_SENTENCE_CAP = {
    "normal": 5,
    "short": 2,
    "extended": 8,
}
CHECK_QUESTION_PATTERNS = (
    re.compile(r"^czy\s+to\s+wyjasnienie\s+jest\s+dla\s+ciebie\s+zrozumiale$"),
    re.compile(r"^czy\s+to\s+jest\s+dla\s+ciebie\s+zrozumiale$"),
    re.compile(r"^czy\s+to\s+dla\s+ciebie\s+jasne$"),
    re.compile(r"^czy\s+to\s+jest\s+jasne$"),
    re.compile(r"^czy\s+zrozumial(?:es|as)?(?:\s+to)?$"),
    re.compile(r"^czy\s+rozumiesz(?:\s+to)?$"),
)


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

        raw_preferences: dict[str, object] = {}
        if hasattr(self.persistence.thread_state, "get_preferences"):
            raw_preferences = self.persistence.thread_state.get_preferences(thread_id)
        thread_preferences = _normalize_thread_preferences(raw_preferences)
        intent = _classify_intent(user_text)
        if intent == "META":
            meta_response, updated_preferences = _handle_meta_request(
                user_text=user_text,
                preferences=thread_preferences,
            )
            if updated_preferences is not None:
                thread_preferences = updated_preferences
                if hasattr(self.persistence.thread_state, "upsert_preferences"):
                    self.persistence.thread_state.upsert_preferences(
                        thread_id=thread_id,
                        preferences=thread_preferences,
                    )

            token_count = estimate_tokens(meta_response)
            fingerprint = self.config.fingerprint()
            fingerprint["llm"]["temperature"] = decode_temperature
            fingerprint["llm"]["top_p"] = decode_top_p
            if seed is not None:
                fingerprint["llm"]["seed"] = seed
            fingerprint["retrieval"] = []
            fingerprint["retrieval_summary"] = {
                "intent": "META",
                "retrieved_count": 0,
                "best_score": 0.0,
                "threshold": self.config.similarity_threshold,
                "has_context": False,
                "candidate_count": 0,
                "main_keyword": None,
                "phrase_norm": None,
                "query_evidence": False,
                "lexical_fallback_used": False,
                "numeric_guard_triggered": False,
            }

            self.persistence.messages.add_assistant_message(
                thread_id=thread_id,
                content=meta_response,
                model=self.config.model_name,
                token_count=token_count,
                sources=[],
                config_fingerprint=fingerprint,
            )
            return AssistantReply(
                content=meta_response,
                sources=[],
                token_count=token_count,
                config_fingerprint=fingerprint,
            )

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
        summary_followup = _is_followup_summary_request(user_text)
        history_source_ids = _history_source_chunk_ids_ordered(
            history_messages=history_before_current,
            n_turns=self.config.n_turns,
        )
        followup_has_history_evidence = bool(history_source_ids)
        ambiguous_followup_without_history = summary_followup and not followup_has_history_evidence
        has_context = retrieved_count > 0 and best_score >= self.config.similarity_threshold
        if ambiguous_followup_without_history or (
            has_context and not retrieval.query_evidence and not summary_followup
        ):
            has_context = False
        elif (
            not has_context
            and summary_followup
            and retrieved_count > 0
            and _has_followup_history_evidence(
                history_messages=history_before_current,
                user_text=user_text,
                n_turns=self.config.n_turns,
            )
        ):
            has_context = True

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
            if ambiguous_followup_without_history:
                content = _build_followup_without_history_response(user_text)
            else:
                content = build_no_context_response()
            citations = []
            numeric_guard_triggered = False
        else:
            followup_focus_terms = (
                _build_followup_focus_entity_terms(
                    user_text=user_text,
                    phrase_norm=retrieval.phrase_norm,
                    main_keyword=retrieval.main_keyword,
                    keywords=retrieval.keywords,
                )
                if summary_followup
                else []
            )
            context_chunks = _select_context_chunks(
                retrieved_chunks=retrieved_chunks,
                phrase_norm=retrieval.phrase_norm,
                main_keyword=retrieval.main_keyword,
                keywords=retrieval.keywords,
            )[:MAX_CONTEXT_CHUNKS]
            if summary_followup and followup_focus_terms:
                context_chunks = _sort_chunks_by_entity_density(
                    context_chunks,
                    followup_focus_terms,
                )[:MAX_CONTEXT_CHUNKS]
            if summary_followup and followup_has_history_evidence:
                history_context_chunks = self._resolve_history_context_chunks(
                    history_source_ids=history_source_ids,
                    retrieved_chunks=retrieved_chunks,
                )
                if history_context_chunks:
                    context_chunks = _prefer_followup_history_context(
                        history_chunks=history_context_chunks,
                        fresh_chunks=context_chunks,
                        focus_terms=followup_focus_terms,
                    )[:MAX_CONTEXT_CHUNKS]
            if summary_followup and has_context and not context_chunks:
                followup_candidates = retrieved_chunks
                search_terms = _build_followup_search_terms(
                    user_text=user_text,
                    history_messages=history_before_current,
                    n_turns=self.config.n_turns,
                )
                if search_terms and hasattr(self.retriever, "retrieve_by_document_contains"):
                    lexical_followup = self._lexical_fallback(
                        phrase_terms=[],
                        search_terms=search_terms,
                        limit=max(self.config.top_k * 5, 25),
                    )
                    if lexical_followup:
                        followup_candidates = _merge_chunks(retrieved_chunks, lexical_followup)
                context_chunks = _select_followup_context_chunks(
                    retrieved_chunks=followup_candidates,
                    history_messages=history_before_current,
                    user_text=user_text,
                    n_turns=self.config.n_turns,
                )[:MAX_CONTEXT_CHUNKS]
            if summary_followup and followup_focus_terms:
                context_chunks = _sort_chunks_by_entity_density(
                    context_chunks,
                    followup_focus_terms,
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
                    user_preferences=_preferences_prompt_block(thread_preferences),
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
                    numeric_guard_triggered = True
                    if summary_followup:
                        content, citations = _build_extractive_fallback(
                            context_chunks=context_chunks,
                            phrase_norm=retrieval.phrase_norm,
                            main_keyword=retrieval.main_keyword,
                            keywords=retrieval.keywords,
                        )
                    else:
                        has_context = False
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

        if has_context and not _looks_like_no_context_response(content):
            content = _apply_content_preferences(
                content=content,
                citations=citations,
                preferences=thread_preferences,
            )

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
            "intent": "CONTENT",
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

    def _resolve_history_context_chunks(
        self,
        *,
        history_source_ids: list[str],
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if not history_source_ids:
            return []

        by_chunk_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
        selected: list[RetrievedChunk] = []
        selected_ids: set[str] = set()
        for chunk_id in history_source_ids:
            chunk = by_chunk_id.get(chunk_id)
            if chunk is None:
                continue
            selected.append(chunk)
            selected_ids.add(chunk_id)

        missing_ids = [chunk_id for chunk_id in history_source_ids if chunk_id not in selected_ids]
        if missing_ids and hasattr(self.retriever, "retrieve_by_chunk_ids"):
            fetched = self.retriever.retrieve_by_chunk_ids(chunk_ids=missing_ids)
            fetched_map = {chunk.chunk_id: chunk for chunk in fetched}
            for chunk_id in missing_ids:
                chunk = fetched_map.get(chunk_id)
                if chunk is None:
                    continue
                selected.append(chunk)

        return selected

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


def _classify_intent(user_text: str) -> str:
    if _is_followup_summary_request(user_text):
        return "CONTENT"

    normalized = normalize_for_match(user_text)
    if any(marker in normalized for marker in META_INTENT_MARKERS):
        return "META"
    return "CONTENT"


def _normalize_thread_preferences(preferences: dict[str, object] | None) -> dict[str, object]:
    payload = preferences or {}
    max_sentences: int | None = None
    raw_max = payload.get("max_sentences")
    if isinstance(raw_max, int) and raw_max > 0:
        max_sentences = min(raw_max, 6)

    ask_check_question = True
    raw_ask_check = payload.get("ask_check_question")
    if isinstance(raw_ask_check, bool):
        ask_check_question = raw_ask_check

    style_short = False
    raw_style_short = payload.get("style_short")
    if isinstance(raw_style_short, bool):
        style_short = raw_style_short

    answer_style = DEFAULT_ANSWER_STYLE
    raw_answer_style = payload.get("answer_style")
    if isinstance(raw_answer_style, str):
        normalized_style = raw_answer_style.strip().lower()
        if normalized_style in ANSWER_STYLE_VALUES:
            answer_style = normalized_style
    elif style_short:
        answer_style = "short"

    return {
        "max_sentences": max_sentences,
        "ask_check_question": ask_check_question,
        "style_short": style_short,
        "answer_style": answer_style,
    }


def _handle_meta_request(
    *,
    user_text: str,
    preferences: dict[str, object],
) -> tuple[str, dict[str, object] | None]:
    normalized = normalize_for_match(user_text)
    if _is_meta_preferences_query(normalized):
        return _format_preferences_response(preferences), None

    updates: dict[str, object] = {}
    max_sentences = _extract_max_sentences(normalized)
    answer_style = _extract_answer_style(normalized)
    if answer_style is not None:
        updates["answer_style"] = answer_style
        updates["style_short"] = answer_style == "short"

    if max_sentences is not None:
        updates["max_sentences"] = max_sentences
        if "answer_style" not in updates:
            updates["answer_style"] = "short"
            updates["style_short"] = True
    elif "bez limitu" in normalized or "bez ograniczen" in normalized:
        updates["max_sentences"] = None

    if any(marker in normalized for marker in DISABLE_CHECK_QUESTION_MARKERS):
        updates["ask_check_question"] = False
    elif any(marker in normalized for marker in ENABLE_CHECK_QUESTION_MARKERS):
        updates["ask_check_question"] = True

    if "krotko" in normalized and "style_short" not in updates:
        updates["style_short"] = True
        if "answer_style" not in updates:
            updates["answer_style"] = "short"

    if not updates:
        return (
            "Rozumiem. Mogę ustawić długość odpowiedzi albo włączyć/wyłączyć pytanie kontrolne.",
            None,
        )

    merged = dict(preferences)
    merged.update(updates)
    normalized_preferences = _normalize_thread_preferences(merged)
    return _build_meta_update_response(updates, normalized_preferences), normalized_preferences


def _build_meta_update_response(
    updates: dict[str, object],
    preferences: dict[str, object],
) -> str:
    lines = ["Ustawienia zapisane."]
    if "max_sentences" in updates:
        limit = preferences.get("max_sentences")
        if isinstance(limit, int):
            lines.append(f"- Maksymalna długość odpowiedzi: {limit} zdania.")
        else:
            lines.append("- Maksymalna długość odpowiedzi: bez limitu.")
    if "ask_check_question" in updates:
        ask = preferences.get("ask_check_question") is True
        lines.append(
            "- Pytanie kontrolne: "
            + ("włączone." if ask else "wyłączone.")
        )
    if "answer_style" in updates:
        lines.append(f"- Styl odpowiedzi: {_format_answer_style(preferences.get('answer_style'))}.")
    return "\n".join(lines)


def _format_preferences_response(preferences: dict[str, object]) -> str:
    limit = preferences.get("max_sentences")
    ask = preferences.get("ask_check_question") is True
    answer_style = _format_answer_style(preferences.get("answer_style"))
    limit_text = f"{limit} zdania" if isinstance(limit, int) else "bez limitu"
    ask_text = "włączone" if ask else "wyłączone"
    return (
        "Aktualne preferencje:\n"
        f"- Maksymalna długość odpowiedzi: {limit_text}.\n"
        f"- Pytanie kontrolne: {ask_text}.\n"
        f"- Styl odpowiedzi: {answer_style}."
    )


def _is_meta_preferences_query(normalized_text: str) -> bool:
    if any(marker in normalized_text for marker in META_QUERY_MARKERS):
        return True
    if "preferencj" in normalized_text and "jak" in normalized_text:
        return True
    if "styl odpowiedzi" in normalized_text and "jak" in normalized_text:
        return True
    return "dlugosc odpowiedzi" in normalized_text and "jak" in normalized_text


def _extract_answer_style(normalized_text: str) -> str | None:
    normalized_plain = re.sub(r"[^a-z0-9\s]", " ", normalized_text)
    normalized_plain = re.sub(r"\s+", " ", normalized_plain).strip()
    if any(
        marker in normalized_plain
        for marker in (
            "tryb krotki",
            "odpowiadaj krotko",
            "styl odpowiedzi short",
            "styl odpowiedzi krotki",
            "ustaw styl odpowiedzi short",
            "odpowiadaj zwiezle",
        )
    ):
        return "short"
    if any(
        marker in normalized_plain
        for marker in (
            "tryb rozszerzony",
            "odpowiadaj szeroko",
            "odpowiadaj rozbudowanie",
            "styl odpowiedzi extended",
            "styl odpowiedzi rozszerzony",
            "ustaw styl odpowiedzi extended",
        )
    ):
        return "extended"
    if any(
        marker in normalized_plain
        for marker in (
            "tryb normalny",
            "odpowiadaj normalnie",
            "styl odpowiedzi normal",
            "ustaw styl odpowiedzi normal",
        )
    ):
        return "normal"
    return None


def _format_answer_style(raw_style: object) -> str:
    normalized = raw_style.strip().lower() if isinstance(raw_style, str) else DEFAULT_ANSWER_STYLE
    if normalized == "short":
        return "krótki"
    if normalized == "extended":
        return "rozszerzony"
    return "normalny"


def _extract_max_sentences(normalized_text: str) -> int | None:
    patterns = (
        r"(?:maks|max)\s*(\d+)\s*zdan",
        r"w\s*(\d+)\s*zdaniach",
        r"(\d+)\s*zdania",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized_text)
        if match is None:
            continue
        value = int(match.group(1))
        if value > 0:
            return min(value, 6)
    return None


def _preferences_prompt_block(preferences: dict[str, object]) -> str | None:
    lines: list[str] = []
    max_sentences = preferences.get("max_sentences")
    ask_check_question = preferences.get("ask_check_question")
    answer_style = _resolve_answer_style(preferences)

    if isinstance(max_sentences, int):
        lines.append(f"- Limit długości odpowiedzi: maksymalnie {max_sentences} zdania.")
    if ask_check_question is False:
        lines.append("- Nie zadawaj pytania sprawdzającego na końcu odpowiedzi.")
    if answer_style == "short" and not isinstance(max_sentences, int):
        lines.append("- Odpowiadaj krótko i konkretnie.")
    elif answer_style == "extended":
        lines.append("- Odpowiadaj szerzej, ale bez dodawania nowych faktów.")
    if not lines:
        return None
    return "\n".join(lines)


def _apply_content_preferences(
    *,
    content: str,
    citations: list[SourceCitation],
    preferences: dict[str, object],
) -> str:
    text_body, citation_suffix = _split_citation_suffix(content) if citations else (content, "")
    answer_style = _resolve_answer_style(preferences)
    ask_check_question = preferences.get("ask_check_question") is not False

    answer_body, generated_question = _split_answer_and_check_question(text_body)
    answer_body = _strip_trailing_check_question(answer_body or text_body)
    if not answer_body:
        answer_body = " ".join(text_body.split()).strip()

    fact_limit = _determine_fact_sentence_limit(
        answer_style=answer_style,
        max_sentences=preferences.get("max_sentences"),
    )
    if fact_limit > 0:
        answer_body = _limit_to_sentences(answer_body, fact_limit)

    if answer_style == "short":
        compact = answer_body.strip() or " ".join(text_body.split()).strip()
        return _join_body_with_citations(compact, citation_suffix)

    sections: list[str] = []
    factual_section = answer_body.strip() or " ".join(text_body.split()).strip()
    if factual_section:
        sections.append(f"Odpowiedź (z materiału):\n{factual_section}")

    learning_section = _build_learning_section(
        answer_style=answer_style,
        max_sentences=preferences.get("max_sentences"),
    )
    if learning_section:
        sections.append(f"Jak to zapamiętać:\n{learning_section}")

    if ask_check_question:
        question = generated_question.strip()
        if not question.endswith("?") or _is_check_question_sentence(question):
            question = _default_check_question_for_style(answer_style)
        sections.append(f"Sprawdź się:\n{question}")

    rewritten = "\n\n".join(section for section in sections if section.strip()).strip()
    if not rewritten:
        rewritten = factual_section
    return _join_body_with_citations(rewritten, citation_suffix)


def _join_body_with_citations(body: str, citation_suffix: str) -> str:
    if not citation_suffix:
        return body
    normalized_suffix = citation_suffix.lstrip("\n")
    if not body:
        return normalized_suffix
    return f"{body}\n\n{normalized_suffix}"


def _resolve_answer_style(preferences: dict[str, object]) -> str:
    raw_style = preferences.get("answer_style")
    if isinstance(raw_style, str):
        normalized = raw_style.strip().lower()
        if normalized in ANSWER_STYLE_VALUES:
            return normalized

    if preferences.get("style_short") is True:
        return "short"
    return DEFAULT_ANSWER_STYLE


def _determine_fact_sentence_limit(*, answer_style: str, max_sentences: object) -> int:
    style_cap = STYLE_FACT_SENTENCE_CAP.get(answer_style, STYLE_FACT_SENTENCE_CAP["normal"])
    if isinstance(max_sentences, int) and max_sentences > 0:
        return min(style_cap, max_sentences)
    return style_cap


def _build_learning_section(*, answer_style: str, max_sentences: object) -> str:
    if answer_style == "extended":
        return (
            "Najpierw nazwij główną myśl własnymi słowami. "
            "Następnie powiąż ją z przyczyną i skutkiem opisanymi w materiale. "
            "Potem sprawdź, czy umiesz streścić temat jednym krótkim wyjaśnieniem. "
            "Na końcu porównaj swoje streszczenie z odpowiedzią i popraw to, co jest nieprecyzyjne."
        )

    if isinstance(max_sentences, int) and max_sentences <= 2:
        return "Powtórz własnymi słowami najważniejszą myśl i wskaż jej sens w kontekście tematu."
    return (
        "Powtórz własnymi słowami główną myśl z odpowiedzi. "
        "Następnie połącz ją z przyczyną i skutkiem opisanymi w materiale."
    )


def _default_check_question_for_style(answer_style: str) -> str:
    if answer_style == "extended":
        return (
            "Jak własnymi słowami wyjaśnisz główną myśl i jej skutek opisany w materiale?"
        )
    return "Jak własnymi słowami wyjaśnisz najważniejszą myśl z tej odpowiedzi?"


def _split_citation_suffix(content: str) -> tuple[str, str]:
    match = re.search(r"\n\[(?:Źródło|Zrodlo):", content)
    if match is None:
        return content.strip(), ""
    start = match.start()
    return content[:start].strip(), content[start:]


def _split_answer_and_check_question(content: str) -> tuple[str, str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if lines and lines[-1].endswith("?"):
        body = "\n".join(lines[:-1]).strip()
        return body, lines[-1]

    compact = " ".join(content.split()).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", compact) if part.strip()]
    if len(sentences) >= 2 and sentences[-1].endswith("?"):
        return " ".join(sentences[:-1]).strip(), sentences[-1]
    return compact, ""


def _strip_trailing_check_question(content: str) -> str:
    compact = " ".join(content.split()).strip()
    if not compact:
        return compact

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", compact) if part.strip()]
    if not sentences:
        return compact

    for _ in range(2):
        if not sentences:
            break
        if _is_check_question_sentence(sentences[-1]):
            sentences.pop()
            continue
        break

    if not sentences:
        return ""
    return " ".join(sentences).strip()


def _is_check_question_sentence(sentence: str) -> bool:
    normalized = normalize_for_match(sentence)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized.startswith("czy "):
        return False
    return any(pattern.fullmatch(normalized) for pattern in CHECK_QUESTION_PATTERNS)


def _limit_to_sentences(content: str, limit: int) -> str:
    if limit <= 0:
        return ""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", content) if part.strip()]
    if not sentences:
        return content.strip()
    selected = sentences[:limit]
    trimmed = " ".join(selected).strip()
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed


def _is_followup_summary_request(user_text: str) -> bool:
    normalized = normalize_for_match(user_text)
    return any(marker in normalized for marker in SUMMARY_FOLLOWUP_MARKERS)


def _build_followup_without_history_response(user_text: str) -> str:
    prefix = "Nie wiem na podstawie dostarczonych materiałów."
    normalized = normalize_for_match(user_text)
    if "wojciech" in normalized:
        return (
            f"{prefix}\n\n"
            "Nie wiem, o którego „Wojciecha” chodzi na podstawie tej rozmowy. "
            "Czy masz na myśli Świętego Wojciecha (biskupa i misjonarza), "
            "czy innego Wojciecha (np. Wojciecha Kossaka)?\n"
            "Doprecyzuj proszę temat jednym zdaniem, a przygotuję podsumowanie."
        )
    return (
        f"{prefix}\n\n"
        "Nie mam w tej rozmowie wcześniejszego kontekstu, do którego mogę wrócić. "
        "Doprecyzuj proszę, jakiej postaci lub wydarzenia dotyczy podsumowanie."
    )


def _has_followup_history_evidence(
    *,
    history_messages: list[ChatMessage],
    user_text: str,
    n_turns: int,
) -> bool:
    if not history_messages:
        return False
    if not _history_source_chunk_ids_ordered(history_messages=history_messages, n_turns=n_turns):
        return False

    topic_terms = _extract_followup_topic_terms(user_text)
    if not topic_terms:
        return False

    recent_messages = history_messages[-max(2, n_turns * 2) :]
    history_text = normalize_for_match("\n".join(msg.content for msg in recent_messages))
    if not history_text:
        return False

    history_tokens = re.findall(r"[a-z0-9]+", history_text)
    return any(_history_contains_topic_term(history_tokens, term) for term in topic_terms)


def _extract_followup_topic_terms(user_text: str) -> list[str]:
    normalized = normalize_for_match(user_text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    topic_terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token.isdigit() or len(token) < 4:
            continue
        if token in FOLLOWUP_TOPIC_STOPWORDS:
            continue
        if token in seen:
            continue
        topic_terms.append(token)
        seen.add(token)
    return topic_terms


def _build_followup_search_terms(
    *,
    user_text: str,
    history_messages: list[ChatMessage],
    n_turns: int,
) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    for term in _extract_followup_topic_terms(user_text):
        for variant in _topic_search_variants(term):
            if variant in seen:
                continue
            terms.append(variant)
            seen.add(variant)

    recent_messages = history_messages[-max(2, n_turns * 2) :]
    for msg in recent_messages:
        if msg.role != "assistant" or not msg.sources:
            continue
        for term in _extract_followup_topic_terms(msg.content):
            for variant in _topic_search_variants(term):
                if variant in seen:
                    continue
                terms.append(variant)
                seen.add(variant)
            if len(terms) >= 10:
                return terms

    return terms


def _topic_search_variants(term: str) -> list[str]:
    variants = [term]
    if len(term) > 4 and term.endswith("u"):
        variants.append(term[:-1])
    if len(term) > 5 and term.endswith("owi"):
        variants.append(term[:-3])
    if len(term) > 5 and term.endswith("em"):
        variants.append(term[:-2])
    output: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if not variant or variant in seen:
            continue
        output.append(variant)
        seen.add(variant)
    return output


def _history_source_chunk_ids_ordered(
    *, history_messages: list[ChatMessage], n_turns: int, limit: int = 12
) -> list[str]:
    recent_messages = history_messages[-max(2, n_turns * 2) :]
    source_ids: list[str] = []
    seen: set[str] = set()
    for msg in reversed(recent_messages):
        if msg.role != "assistant":
            continue
        for source in msg.sources:
            if source.chunk_id in seen:
                continue
            source_ids.append(source.chunk_id)
            seen.add(source.chunk_id)
            if len(source_ids) >= limit:
                return source_ids
    return source_ids


def _select_followup_context_chunks(
    *,
    retrieved_chunks: list[RetrievedChunk],
    history_messages: list[ChatMessage],
    user_text: str,
    n_turns: int,
) -> list[RetrievedChunk]:
    if not retrieved_chunks:
        return []

    source_ids = _history_source_chunk_ids_ordered(
        history_messages=history_messages,
        n_turns=n_turns,
    )
    if source_ids:
        by_history_sources = [chunk for chunk in retrieved_chunks if chunk.chunk_id in source_ids]
        if by_history_sources:
            return by_history_sources

    topic_terms = _extract_followup_topic_terms(user_text)
    if not topic_terms:
        return retrieved_chunks

    roots = [_topic_token_root(term) for term in topic_terms]
    by_topic_terms: list[RetrievedChunk] = []
    for chunk in retrieved_chunks:
        chunk_tokens = re.findall(r"[a-z0-9]+", normalize_for_match(chunk.text))
        if any(any(token.startswith(root) for token in chunk_tokens) for root in roots):
            by_topic_terms.append(chunk)
    if by_topic_terms:
        return by_topic_terms

    return retrieved_chunks


def _build_followup_focus_entity_terms(
    *,
    user_text: str,
    phrase_norm: str | None,
    main_keyword: str | None,
    keywords: list[str],
) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add_term(raw_term: str | None) -> None:
        if not raw_term:
            return
        normalized = normalize_for_match(raw_term)
        if len(normalized) < 4 or normalized in seen:
            return
        terms.append(normalized)
        seen.add(normalized)

    add_term(phrase_norm)
    add_term(main_keyword)
    for keyword in keywords:
        add_term(keyword)
    for term in _extract_followup_topic_terms(user_text):
        add_term(term)

    normalized_user = normalize_for_match(user_text)
    if "wojciech" in normalized_user:
        for special in (
            "swiety wojciech",
            "sw wojciech",
            "wojciech",
            "wojciecha",
        ):
            add_term(special)

    return terms


def _entity_density_score(text: str, entity_terms: list[str]) -> int:
    normalized_text = normalize_for_match(text)
    if not normalized_text or not entity_terms:
        return 0

    early_text = normalized_text[:200]
    score = 0
    for term in entity_terms:
        if " " in term:
            occurrences = normalized_text.count(term)
            if occurrences:
                score += occurrences * 3
                if term in early_text:
                    score += 2
            continue

        occurrences = _count_root_occurrences(normalized_text, term)
        if occurrences:
            score += occurrences
            if _count_root_occurrences(early_text, term):
                score += 1
    return score


def _count_root_occurrences(text: str, term: str) -> int:
    if not text or not term:
        return 0
    root = _topic_token_root(term)
    tokens = re.findall(r"[a-z0-9]+", text)
    return sum(1 for token in tokens if token.startswith(root))


def _sort_chunks_by_entity_density(
    chunks: list[RetrievedChunk], focus_terms: list[str]
) -> list[RetrievedChunk]:
    if not chunks or not focus_terms:
        return chunks

    return sorted(
        chunks,
        key=lambda chunk: (
            -_entity_density_score(chunk.text, focus_terms),
            -chunk.score,
            chunk.chunk_id,
        ),
    )


def _prefer_followup_history_context(
    *,
    history_chunks: list[RetrievedChunk],
    fresh_chunks: list[RetrievedChunk],
    focus_terms: list[str],
) -> list[RetrievedChunk]:
    ranked_history = _sort_chunks_by_entity_density(history_chunks, focus_terms)
    ranked_fresh = _sort_chunks_by_entity_density(fresh_chunks, focus_terms)
    if not ranked_history:
        return ranked_fresh
    if not ranked_fresh:
        return ranked_history

    history_best = _entity_density_score(ranked_history[0].text, focus_terms)
    fresh_best = _entity_density_score(ranked_fresh[0].text, focus_terms)
    if history_best >= fresh_best:
        return ranked_history
    return ranked_fresh


def _history_contains_topic_term(history_tokens: list[str], term: str) -> bool:
    if not term:
        return False
    if any(term in token for token in history_tokens):
        return True
    root = _topic_token_root(term)
    return any(token.startswith(root) for token in history_tokens)


def _topic_token_root(token: str) -> str:
    if token.isdigit():
        return token
    if len(token) <= 4:
        return token
    return token[: max(4, len(token) - 2)]


def _normalize_dash_variants(text: str) -> str:
    return text.translate(str.maketrans({"–": "-", "—": "-", "−": "-"}))


def _extract_numeric_years(text: str) -> set[str]:
    normalized = _normalize_dash_variants(text)
    return set(re.findall(r"\b\d{3,4}\b", normalized))


def _has_unsupported_numeric_claims(
    generated_text: str,
    retrieved_chunks: list[RetrievedChunk],
) -> bool:
    years = _extract_numeric_years(generated_text)
    if not years:
        return False

    context_text = "\n".join(chunk.text for chunk in retrieved_chunks)
    context_years = _extract_numeric_years(context_text)
    return not years.issubset(context_years)


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
    snippet = _extract_chunk_snippet(
        evidence_chunk.text,
        phrase_norm=phrase_norm,
        main_keyword=main_keyword,
        keywords=keywords,
    )
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


def _extract_chunk_snippet(
    text: str,
    *,
    phrase_norm: str | None,
    main_keyword: str | None,
    keywords: list[str],
) -> str:
    compact = " ".join(text.split()).strip()
    if not compact:
        return "w materiale jest krótka wzmianka na ten temat."

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", compact) if part.strip()]
    if not sentences:
        return "w materiale jest krótka wzmianka na ten temat."

    scored: list[tuple[int, int, int, str]] = []
    for idx, sentence in enumerate(sentences):
        score = _score_sentence(
            sentence=sentence,
            phrase_norm=phrase_norm,
            main_keyword=main_keyword,
            keywords=keywords,
        )
        scored.append((score, len(sentence), idx, sentence))

    ranked = sorted(scored, key=lambda row: (-row[0], -row[1], row[2]))
    selected = [row[3] for row in ranked[:2]]
    if len(selected) == 2:
        index_map = {sentence: idx for idx, sentence in enumerate(sentences)}
        selected = sorted(selected, key=lambda sentence: index_map.get(sentence, 10**6))

    cleaned_selected = [_clean_fallback_sentence(sentence) for sentence in selected]
    cleaned_selected = [sentence for sentence in cleaned_selected if sentence]
    snippet = " ".join(cleaned_selected).strip()
    if not snippet:
        snippet = _clean_fallback_sentence(sentences[0])
    if len(snippet) > 320:
        snippet = snippet[:317].rstrip() + "..."
    if snippet and snippet[-1] not in ".!?":
        snippet += "."
    return snippet


def _score_sentence(
    *,
    sentence: str,
    phrase_norm: str | None,
    main_keyword: str | None,
    keywords: list[str],
) -> int:
    normalized = normalize_for_match(sentence)
    score = 0

    if phrase_norm and phrase_norm in normalized:
        score += 2
    if main_keyword and normalize_for_match(main_keyword) in normalized:
        score += 1
    if len(keywords) >= 2:
        keyword_hits = sum(1 for keyword in keywords if normalize_for_match(keyword) in normalized)
        if keyword_hits >= 2:
            score += 1

    if len(normalized) < 24:
        score -= 1
    if re.match(r"^\s*\d+\s+", sentence):
        score -= 2
    if sentence.endswith(("(", "-", "–", "—")):
        score -= 1
    return score


def _clean_fallback_sentence(sentence: str) -> str:
    cleaned = sentence.strip()
    cleaned = re.sub(r"^\d{1,3}\s+(?=[A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż])", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
