"""Hybrid retrieval helpers (vector candidates + lexical reranking)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.types import RetrievedChunk

POLISH_STOPWORDS = {
    "a",
    "aby",
    "albo",
    "ale",
    "bo",
    "byl",
    "byla",
    "byli",
    "bylo",
    "byc",
    "co",
    "czym",
    "czy",
    "do",
    "dla",
    "i",
    "jak",
    "jaka",
    "jakie",
    "jakim",
    "jakich",
    "jako",
    "jest",
    "kim",
    "kto",
    "ktora",
    "ktorym",
    "na",
    "o",
    "oraz",
    "po",
    "pod",
    "sie",
    "to",
    "w",
    "we",
    "z",
    "za",
    "ze",
}

GENERIC_QUERY_TERMS = {
    "dlaczego",
    "napisz",
    "opowiedz",
    "podaj",
    "pomoz",
    "porownaj",
    "sposob",
    "uporzadkowac",
    "wskaz",
    "wytlumacz",
    "wyjasnij",
    "znaczenie",
}

PREFERRED_TOPIC_PAIRS: dict[tuple[str, str], str] = {
    ("zjazd", "gniezniensk"): "zjazd gnieznienski",
    ("boleslaw", "chrobr"): "boleslaw chrobry",
    ("swiety", "wojciech"): "swiety wojciech",
    ("metropoli", "gniezniensk"): "metropolia gnieznienska",
}

POLISH_ASCII_TRANSLATION = str.maketrans(
    {
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
        "Ą": "A",
        "Ć": "C",
        "Ę": "E",
        "Ł": "L",
        "Ń": "N",
        "Ó": "O",
        "Ś": "S",
        "Ź": "Z",
        "Ż": "Z",
    }
)


@dataclass(slots=True)
class KeywordQuery:
    keywords: list[str]
    main_keyword: str | None
    phrase_norm: str | None
    phrase_terms: list[str]
    lexical_terms: list[str]


@dataclass(slots=True)
class _PairCandidate:
    phrase: str
    score: int


def analyze_query_keywords(query: str) -> KeywordQuery:
    token_rows = _token_rows(query)
    keyword_rows = [
        (original, normalized)
        for original, normalized in token_rows
        if _should_keep_keyword(normalized)
    ]
    keywords = _unique_preserve([normalized for _, normalized in keyword_rows])
    keyword_to_original = _keyword_to_original(keyword_rows)
    lexical_terms = _build_lexical_terms(keyword_rows)

    focus_phrase = _extract_focus_entity_phrase(query)
    phrase_candidates = _build_phrase_candidates(
        keyword_rows=keyword_rows,
        keyword_to_original=keyword_to_original,
        focus_phrase=focus_phrase,
    )

    phrase_norm = normalize_for_match(phrase_candidates[0]) if phrase_candidates else None
    phrase_terms = _build_phrase_terms(phrase_candidates, focus_phrase)
    main_keyword = _choose_main_keyword(keywords)
    return KeywordQuery(
        keywords=keywords,
        main_keyword=main_keyword,
        phrase_norm=phrase_norm,
        phrase_terms=phrase_terms,
        lexical_terms=lexical_terms,
    )


def rerank_chunks(
    chunks: list[RetrievedChunk], query: str
) -> tuple[list[RetrievedChunk], KeywordQuery]:
    keyword_query = analyze_query_keywords(query)
    reranked = rerank_chunks_with_keyword_query(chunks, keyword_query)
    return reranked, keyword_query


def rerank_chunks_with_keyword_query(
    chunks: list[RetrievedChunk], keyword_query: KeywordQuery
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    if not keyword_query.keywords:
        return sorted(chunks, key=lambda chunk: (-chunk.score, chunk.chunk_id))

    def sort_key(chunk: RetrievedChunk) -> tuple[int, int, float, str]:
        phrase_hit = int(chunk_phrase_hit(chunk, keyword_query.phrase_norm))
        keyword_hits = chunk_keyword_hit_count(chunk, keyword_query.keywords)
        return (-phrase_hit, -keyword_hits, -chunk.score, chunk.chunk_id)

    return sorted(chunks, key=sort_key)


def chunk_contains_keyword(chunk: RetrievedChunk, keyword: str | None) -> bool:
    if not keyword:
        return False
    return keyword in normalize_for_match(chunk.text)


def chunk_phrase_hit(chunk: RetrievedChunk, phrase_norm: str | None) -> bool:
    if not phrase_norm:
        return False

    normalized_text = normalize_for_match(chunk.text)
    if phrase_norm in normalized_text:
        return True

    phrase_tokens = [token for token in phrase_norm.split() if token]
    if len(phrase_tokens) < 2:
        return False

    text_tokens = _tokenize_words(normalized_text)
    if not text_tokens:
        return False

    return all(
        any(candidate.startswith(_token_root(token)) for candidate in text_tokens)
        for token in phrase_tokens
    )


def chunk_keyword_hit_count(chunk: RetrievedChunk, keywords: list[str]) -> int:
    if not keywords:
        return 0
    normalized_text = normalize_for_match(chunk.text)
    return sum(1 for keyword in keywords if keyword in normalized_text)


def should_use_lexical_fallback(chunks: list[RetrievedChunk], keyword_query: KeywordQuery) -> bool:
    if not chunks:
        return bool(keyword_query.keywords or keyword_query.phrase_terms)
    return not has_required_evidence(chunks, keyword_query)


def has_query_evidence(chunks: list[RetrievedChunk], keyword_query: KeywordQuery) -> bool:
    return has_required_evidence(chunks, keyword_query)


def has_required_evidence(chunks: list[RetrievedChunk], keyword_query: KeywordQuery) -> bool:
    if not chunks:
        return False
    if keyword_query.phrase_norm:
        return any(chunk_phrase_hit(chunk, keyword_query.phrase_norm) for chunk in chunks)
    if len(keyword_query.keywords) >= 2:
        return any(
            chunk_keyword_hit_count(chunk, keyword_query.keywords) >= 2 for chunk in chunks
        )
    if keyword_query.main_keyword:
        return any(chunk_contains_keyword(chunk, keyword_query.main_keyword) for chunk in chunks)
    return True


def ensure_top_k_contains_evidence(
    chunks: list[RetrievedChunk],
    keyword_query: KeywordQuery,
    top_k: int,
) -> list[RetrievedChunk]:
    if top_k <= 0:
        return []
    selected = list(chunks[:top_k])
    if not selected:
        return selected
    if has_required_evidence(selected, keyword_query):
        return selected

    evidence = find_first_evidence_chunk(chunks, keyword_query)
    if evidence is None:
        return selected
    if any(chunk.chunk_id == evidence.chunk_id for chunk in selected):
        return selected

    remaining = [chunk for chunk in selected if chunk.chunk_id != evidence.chunk_id]
    return [evidence, *remaining][:top_k]


def find_first_evidence_chunk(
    chunks: list[RetrievedChunk],
    keyword_query: KeywordQuery,
) -> RetrievedChunk | None:
    if keyword_query.phrase_norm:
        for chunk in chunks:
            if chunk_phrase_hit(chunk, keyword_query.phrase_norm):
                return chunk
        return None

    if len(keyword_query.keywords) >= 2:
        for chunk in chunks:
            if chunk_keyword_hit_count(chunk, keyword_query.keywords) >= 2:
                return chunk
        return None

    if keyword_query.main_keyword:
        for chunk in chunks:
            if chunk_contains_keyword(chunk, keyword_query.main_keyword):
                return chunk
        return None
    return None


def normalize_for_match(text: str) -> str:
    if not text:
        return ""
    translated = text.translate(POLISH_ASCII_TRANSLATION)
    decomposed = unicodedata.normalize("NFKD", translated)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower().strip()


def _token_rows(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for token in _tokenize_words(text):
        normalized = normalize_for_match(token)
        if not normalized:
            continue
        rows.append((token, normalized))
    return rows


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9ĄąĆćĘęŁłŃńÓóŚśŹźŻż]+", text)


def _token_root(token: str) -> str:
    if token.isdigit():
        return token
    if len(token) <= 4:
        return token
    return token[: max(4, len(token) - 2)]


def _should_keep_keyword(token: str) -> bool:
    if len(token) <= 2:
        return False
    return token not in POLISH_STOPWORDS


def _keyword_to_original(keyword_rows: list[tuple[str, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for original, normalized in keyword_rows:
        mapping.setdefault(normalized, original)
    return mapping


def _build_lexical_terms(keyword_rows: list[tuple[str, str]]) -> list[str]:
    lexical_terms: list[str] = []
    seen: set[str] = set()
    for original, _ in keyword_rows:
        for term in _lexical_variants(original):
            cleaned = term.strip()
            if not cleaned or cleaned in seen:
                continue
            lexical_terms.append(cleaned)
            seen.add(cleaned)
    return lexical_terms


def _build_phrase_candidates(
    *,
    keyword_rows: list[tuple[str, str]],
    keyword_to_original: dict[str, str],
    focus_phrase: str | None,
) -> list[str]:
    candidates: list[str] = []
    if focus_phrase:
        candidates.append(focus_phrase)

    pair_candidates = _rank_pair_candidates(keyword_rows, keyword_to_original)
    for candidate in pair_candidates:
        if candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) >= 3:
            break
    return candidates


def _rank_pair_candidates(
    keyword_rows: list[tuple[str, str]],
    keyword_to_original: dict[str, str],
) -> list[str]:
    ordered_keywords = _unique_preserve([normalized for _, normalized in keyword_rows])
    if len(ordered_keywords) < 2:
        return []

    candidates: list[_PairCandidate] = []
    seen_pairs: set[tuple[str, str]] = set()

    for idx in range(len(ordered_keywords) - 1):
        left = ordered_keywords[idx]
        right = ordered_keywords[idx + 1]
        if left == right:
            continue
        pair = (left, right)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        candidates.append(_build_pair_candidate(left, right, keyword_to_original, adjacent=True))

    informative = [kw for kw in ordered_keywords if not _is_generic_keyword(kw)]
    informative = informative[:5]
    for idx, left in enumerate(informative):
        for right in informative[idx + 1 :]:
            pair = (left, right)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            candidates.append(
                _build_pair_candidate(
                    left,
                    right,
                    keyword_to_original,
                    adjacent=False,
                )
            )

    ranked = sorted(candidates, key=lambda item: (-item.score, item.phrase))
    phrases = _unique_preserve([item.phrase for item in ranked])
    return phrases


def _build_pair_candidate(
    left: str,
    right: str,
    keyword_to_original: dict[str, str],
    *,
    adjacent: bool,
) -> _PairCandidate:
    canonical = _canonical_phrase_for_pair(left, right)
    left_original = keyword_to_original.get(left, left)
    right_original = keyword_to_original.get(right, right)
    phrase = canonical or f"{left_original} {right_original}"
    score = 0
    if canonical:
        score += 200
    if adjacent:
        score += 20
    if not _is_generic_keyword(left):
        score += 15
    if not _is_generic_keyword(right):
        score += 15
    score += len(left) + len(right)
    if _is_topic_keyword(left):
        score += 10
    if _is_topic_keyword(right):
        score += 10
    return _PairCandidate(phrase=phrase, score=score)


def _extract_focus_entity_phrase(query: str) -> str | None:
    patterns = [
        r"\bkim\s+by[łl]\s+(.+?)(?:[?.!]|$)",
        r"\bkto\s+(?:to\s+)?by[łl]\s+(.+?)(?:[?.!]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        phrase = match.group(1).strip()
        tokens = _tokenize_words(phrase)
        if not tokens:
            continue
        filtered = [token for token in tokens if _should_keep_keyword(normalize_for_match(token))]
        if not filtered:
            continue
        cleaned = " ".join(filtered[:3])
        canonical = _canonical_phrase_for_text(cleaned)
        return canonical or cleaned
    return None


def _canonical_phrase_for_text(text: str) -> str | None:
    tokens = [normalize_for_match(token) for token in _tokenize_words(text)]
    if len(tokens) < 2:
        return None
    return _canonical_phrase_for_pair(tokens[0], tokens[1])


def _build_phrase_terms(phrase_candidates: list[str], focus_phrase: str | None) -> list[str]:
    terms: list[str] = []
    for phrase in phrase_candidates[:3]:
        terms.extend(_phrase_variants(phrase))

    if focus_phrase:
        for token in _tokenize_words(focus_phrase):
            terms.extend(_lexical_variants(token))

    return _unique_preserve([term for term in terms if term.strip()])


def _canonical_phrase_for_pair(left: str, right: str) -> str | None:
    for (left_root, right_root), canonical in PREFERRED_TOPIC_PAIRS.items():
        if _matches_roots(left, right, left_root, right_root):
            return canonical
    return None


def _matches_roots(
    left: str,
    right: str,
    left_root: str,
    right_root: str,
) -> bool:
    return (
        left.startswith(left_root)
        and right.startswith(right_root)
    ) or (
        left.startswith(right_root)
        and right.startswith(left_root)
    )


def _choose_main_keyword(keywords: list[str]) -> str | None:
    if not keywords:
        return None

    def main_score(token: str) -> tuple[int, int]:
        generic_penalty = 0 if not _is_generic_keyword(token) else -1
        topic_bonus = 1 if _is_topic_keyword(token) else 0
        return (generic_penalty + topic_bonus, len(token))

    return sorted(keywords, key=lambda token: main_score(token), reverse=True)[0]


def _is_generic_keyword(token: str) -> bool:
    if token in GENERIC_QUERY_TERMS:
        return True
    generic_prefixes = (
        "wyjasn",
        "wytlumacz",
        "pomoz",
        "napisz",
        "podaj",
        "opowiedz",
        "wskaz",
        "porown",
    )
    return token.startswith(generic_prefixes)


def _is_topic_keyword(token: str) -> bool:
    roots = {
        "zjazd",
        "gniezniensk",
        "swiety",
        "wojciech",
        "boleslaw",
        "chrobr",
        "metropoli",
        "mieszk",
    }
    return any(token.startswith(root) for root in roots)


def _lexical_variants(token: str) -> list[str]:
    base = token.strip()
    base_ascii = _strip_diacritics(base)
    return [
        base,
        base.lower(),
        base_ascii,
        base_ascii.lower(),
        base_ascii.capitalize(),
    ]


def _strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _phrase_variants(phrase: str) -> list[str]:
    ascii_phrase = _strip_diacritics(phrase)
    variants = [
        phrase,
        ascii_phrase,
        phrase.lower(),
        ascii_phrase.lower(),
    ]
    return _unique_preserve([variant.strip() for variant in variants if variant.strip()])


def _unique_preserve(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        output.append(value)
        seen.add(value)
    return output
