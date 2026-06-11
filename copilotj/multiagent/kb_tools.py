# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
import shutil
import tomllib
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from copilotj.core.config import get_home, load_config
from copilotj.core.message import TextMessage
from copilotj.core.model_client import new_model_client

_SOURCE_ROOT = Path(__file__).resolve().parent.parent.parent
_HOME = get_home()
KB_ROOT = _HOME / "knowledge_bank"
KB_TASK = KB_ROOT / "task"
KB_MACRO = KB_ROOT / "macro"
KB_RESEARCH = KB_ROOT / "research"
KB_INDEX = KB_ROOT / "index"
KB_REGISTRY = KB_INDEX / "registry.jsonl"


def _load_macro_plugin_names() -> list[str]:
    """Load all plugin names from macro TOML files."""
    plugin_names = []
    try:
        for macro_file in KB_MACRO.glob("*.toml"):
            try:
                with open(macro_file, "rb") as f:
                    data = tomllib.load(f)
                    if "content" in data and "plugin_name" in data["content"]:
                        plugin_name = data["content"]["plugin_name"]
                        if plugin_name and plugin_name.strip():
                            plugin_names.append(plugin_name.strip())
            except Exception:
                continue  # Skip invalid files
    except Exception:
        pass
    return sorted(list(set(plugin_names)))  # Remove duplicates and sort


def _ensure_dirs() -> None:
    """Ensure knowledge bank directory structure exists."""
    _bootstrap_kb()
    KB_ROOT.mkdir(parents=True, exist_ok=True)
    KB_TASK.mkdir(parents=True, exist_ok=True)
    KB_MACRO.mkdir(parents=True, exist_ok=True)
    KB_RESEARCH.mkdir(parents=True, exist_ok=True)
    KB_INDEX.mkdir(parents=True, exist_ok=True)


def _bootstrap_kb() -> None:
    """Copy knowledge_bank from project source to COPILOTJ_HOME if missing."""
    if KB_ROOT.exists() and any(KB_ROOT.iterdir()):
        return
    source = _SOURCE_ROOT / "knowledge_bank"
    if source.exists():
        shutil.copytree(source, KB_ROOT, dirs_exist_ok=True)


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _slugify(text: str) -> str:
    t = text.strip().lower()
    # Allow underscores, dots, and hyphens, but convert spaces to dots
    t = re.sub(r"[^a-z0-9._-]+", ".", t)
    t = re.sub(r"\.+", ".", t).strip(".")
    return t or "entry"


def _normalize_synonyms(s: str) -> str:
    # Minimal synonyms normalization
    repl = {
        "star dist": "stardist",
        "cell pose": "cellpose",
        "gray scale": "grayscale",
        "8 bit": "8-bit",
        "nuclei": "nuclei",
    }
    s_low = s.lower()
    for k, v in repl.items():
        s_low = s_low.replace(k, v)
    return s_low


def _tokenize(s: str) -> list[str]:
    s = _normalize_synonyms(s)
    return [t for t in re.split(r"[^a-z0-9+-]+", s) if len(t) >= 2]


def _iter_kb_files() -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for t, p in ("task", KB_TASK), ("macro", KB_MACRO), ("research", KB_RESEARCH):
        if p.exists():
            for f in p.glob("*.toml"):
                files.append((t, f))
    return files


def rebuild_registry() -> int:
    _ensure_dirs()
    count = 0
    with KB_REGISTRY.open("w", encoding="utf-8") as out:
        for typ, path in _iter_kb_files():
            try:
                data = _read_toml(path)
                # Extract nested content for macro entries
                content = data.get("content", {})

                rec = {
                    "id": data.get("signature") or _slugify(path.stem),
                    "type": typ,
                    "path": str(path).replace("\\", "/"),
                    "name": data.get("name"),
                    "data_type": data.get("data_type"),
                    "tags": data.get("tags") or [],
                    "topic": data.get("topic") or [],
                    "description": data.get("description") or data.get("summary"),
                    "question": data.get("question")
                    or (f"How to use {data.get('name')} in ImageJ?" if typ == "macro" else None),
                    "summary": data.get("summary"),
                    # Look for tips/requirements in both top-level and content section
                    "tips": data.get("tips") or content.get("tips") or [],
                    "requirements": data.get("requirements")
                    or content.get("requirements")
                    or data.get("requires")
                    or {},
                    "defaults": data.get("defaults") or {},
                    "highlights": data.get("highlights") or [],  # For research entries
                    "references": data.get("references") or [],  # For research entries
                    # Additional macro-specific fields
                    "plugin_name": content.get("plugin_name"),
                    "plugin_description": content.get("plugin_description"),
                    "correct_syntax": content.get("correct_syntax"),
                    "site": content.get("site"),
                    # Workflow steps for tasks
                    "workflow_steps": data.get("steps", []),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
            except Exception:
                # Skip malformed file
                continue
    return count


def _load_registry() -> list[dict[str, Any]]:
    if not KB_REGISTRY.exists():
        rebuild_registry()
    entries: list[dict[str, Any]] = []
    try:
        with KB_REGISTRY.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return entries


def _build_tfidf_vectorizer(entries: list[dict[str, Any]]) -> tuple[TfidfVectorizer, np.ndarray]:
    """Build TF-IDF vectorizer and document matrix using scikit-learn."""
    # Extract all searchable text from entries
    documents = []
    for entry in entries:
        fields = [
            str(entry.get("name") or ""),
            str(entry.get("description") or ""),
            str(entry.get("question") or ""),  # Add question field with high importance
            " ".join(map(str, entry.get("tags") or [])),
            " ".join(map(str, entry.get("topic") or [])),
            str(entry.get("data_type") or ""),
            str(entry.get("id") or ""),
        ]
        text = _normalize_synonyms(" ".join(fields))
        documents.append(text)

    # Create TF-IDF vectorizer with optimized parameters
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,  # Keep domain-specific terms
        max_features=1000,  # Limit vocabulary size for efficiency
        ngram_range=(1, 2),  # Include bigrams for better context
        min_df=1,  # Include rare terms (small corpus)
        max_df=0.9,  # Remove very common terms
        sublinear_tf=True,  # Use sublinear scaling for TF
        norm="l2",  # L2 normalization
    )

    # Fit and transform documents
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix


def _tfidf_similarity(
    query: str, image_desc: str, vectorizer: TfidfVectorizer, tfidf_matrix: np.ndarray, entry_index: int
) -> float:
    """Calculate TF-IDF cosine similarity between query and entry using sklearn."""
    # Combine query and image description
    query_text = _normalize_synonyms(query + (" " + image_desc if image_desc else ""))

    # Transform query to TF-IDF vector
    query_vector = vectorizer.transform([query_text])

    # Get entry vector
    entry_vector = tfidf_matrix[entry_index : entry_index + 1]

    # Calculate cosine similarity
    similarity = cosine_similarity(query_vector, entry_vector)[0][0]
    return float(similarity)


def _enhanced_query_score(
    query: str,
    image_desc: str,
    entry: dict[str, Any],
    topic: str | None = None,
    question: str | None = None,
    vectorizer: TfidfVectorizer | None = None,
    tfidf_matrix: np.ndarray | None = None,
    entry_index: int | None = None,
) -> tuple[float, list[str]]:
    """Enhanced scoring combining multiple signals including perception-based matching."""
    reasons = []
    total_score = 0.0

    # 1. Question matching (highest priority - 35% weight)
    question_score = 0.0
    if question and entry.get("question"):
        q_tokens = _tokenize(question.lower())
        entry_question = str(entry.get("question", "")).lower()
        entry_q_tokens = _tokenize(entry_question)
        if q_tokens and entry_q_tokens:
            overlap = len(set(q_tokens) & set(entry_q_tokens))
            if overlap > 0:
                question_score = overlap / max(len(q_tokens), len(entry_q_tokens))
                reasons.append("question_match")
                total_score += 0.35 * question_score

    # 2. Traditional lexical matching (20% weight - reduced)
    q_tokens = _tokenize(query + (" " + image_desc if image_desc else ""))
    lex_score = _lexical_score(q_tokens, entry)
    if lex_score > 0:
        reasons.append("query_match")
        total_score += 0.2 * lex_score

    # 3. Data type matching for image descriptions (20% weight - reduced)
    dtm_score = _data_type_match(image_desc, entry)
    if dtm_score > 0:
        reasons.append("data_type_match")
        total_score += 0.2 * dtm_score

    # 4. Topic matching (15% weight - reduced)
    topic_score = 0.0
    if topic:
        topic_tokens = _tokenize(topic.lower())
        entry_topics = entry.get("topic", [])
        if entry_topics:
            entry_topic_tokens = set()
            for t in entry_topics:
                entry_topic_tokens.update(_tokenize(str(t).lower()))
            topic_overlap = len(set(topic_tokens) & entry_topic_tokens)
            if topic_overlap > 0:
                topic_score = topic_overlap / len(topic_tokens)
                reasons.append("topic_match")
                total_score += 0.15 * topic_score

    # 5. Enhanced image perception matching (5% weight - reduced)
    perception_score = _perception_match(image_desc, entry)
    if perception_score > 0:
        reasons.append("perception_match")
        total_score += 0.05 * perception_score

    # 6. Tags overlap (3% weight - reduced)
    if entry.get("tags"):
        q_set = set(q_tokens)
        tag_tokens = set()
        for tag in entry.get("tags", []):
            tag_tokens.update(_tokenize(str(tag).lower()))
        overlap = len(q_set & tag_tokens)
        if overlap > 0:
            reasons.append("tags_overlap")
            total_score += 0.03 * (overlap / len(q_tokens))

    # 7. TF-IDF similarity (2% weight - reduced)
    if vectorizer is not None and tfidf_matrix is not None and entry_index is not None:
        tfidf_score = _tfidf_similarity(query, image_desc, vectorizer, tfidf_matrix, entry_index)
        if tfidf_score > 0.1:  # Only count significant similarities
            reasons.append("tfidf_match")
            total_score += 0.02 * tfidf_score

    return total_score, reasons


def _perception_match(image_desc: str, entry: dict[str, Any]) -> float:
    """Enhanced perception-based matching using task TOML data_type field."""
    if not image_desc:
        return 0.0

    # Normalize image description
    desc_tokens = _tokenize(_normalize_synonyms(image_desc.lower()))
    if not desc_tokens:
        return 0.0

    # Primary focus: data_type field from task TOML (most relevant for image description)
    entry_data_type = str(entry.get("data_type") or "")
    data_type_score = 0.0
    data_type_tokens = set()  # Initialize empty set

    if (
        entry_data_type
        and "not specified" not in entry_data_type.lower()
        and "not applicable" not in entry_data_type.lower()
    ):
        # Extract tokens from data_type field
        data_type_tokens = set(_tokenize(_normalize_synonyms(entry_data_type.lower())))

        # Calculate direct token overlap between image description and data_type
        overlap = len(set(desc_tokens) & data_type_tokens)
        if overlap > 0:
            data_type_score = overlap / len(desc_tokens)

    # Secondary matching: enhanced keyword-based matching
    # fmt: off
    perception_keywords = {
        # Cell types and structures
        "cell", "cells", "nuclei", "nucleus", "cytoplasm", "membrane", "organelle", "mitochondria",
        # Image types and techniques
        "fluorescence", "brightfield", "darkfield", "phase", "microscopy", "confocal", "live", "fixed",
        # Image properties
        "8-bit", "16-bit", "32-bit", "rgb", "grayscale", "multichannel", "stack", "timelapse", "z-stack",
        # Staining and markers
        "dapi", "hoechst", "gfp", "rfp", "cfp", "yfp", "fitc", "cy3", "cy5", "alexa", "phalloidin",
        # Analysis contexts
        "bright", "dark", "background", "foreground", "circular", "oval", "spherical", "elongated",
        # Specific imaging conditions
        "dimensions", "pixels", "resolution", "contrast", "intensity", "signal", "noise",
    }
    # fmt: on

    # Extract all relevant text from entry for secondary matching
    searchable_fields = [
        str(entry.get("description") or ""),
        " ".join(map(str, entry.get("topic") or [])),
        " ".join(map(str, entry.get("tags") or [])),
    ]
    entry_text = _normalize_synonyms(" ".join(searchable_fields).lower())
    entry_tokens = set(_tokenize(entry_text))

    # Count keyword matches
    keyword_matches = 0
    for token in desc_tokens:
        if token in perception_keywords and (token in entry_tokens or token in data_type_tokens):
            keyword_matches += 1

    keyword_score = 0.0
    if keyword_matches > 0:
        keyword_score = keyword_matches / len(desc_tokens)

    # Combine scores: prioritize data_type field (70%) + keyword matching (30%)
    final_score = 0.7 * data_type_score + 0.3 * keyword_score

    return min(final_score, 1.0)  # Cap at 1.0


def _lexical_score(q_tokens: list[str], entry: dict[str, Any]) -> float:
    # Aggregate searchable text
    fields = [
        str(entry.get("name") or ""),
        str(entry.get("description") or ""),
        str(entry.get("question") or ""),  # Add question field
        " ".join(map(str, entry.get("tags") or [])),
        " ".join(map(str, entry.get("topic") or [])),
        str(entry.get("data_type") or ""),
        str(entry.get("id") or ""),
    ]
    text = _normalize_synonyms(" ".join(fields))
    score = sum(1 if t in text else 0 for t in q_tokens)
    return score / max(len(q_tokens), 1)


def _data_type_match(image_desc: str, entry: dict[str, Any]) -> float:
    if not image_desc:
        return 0.0

    ed = _normalize_synonyms(image_desc)
    dt = _normalize_synonyms(str(entry.get("data_type") or ""))
    if not dt:
        return 0.0

    tokens = _tokenize(ed)
    if not tokens:
        return 0.0

    hits = sum(1 for t in tokens if t in dt)
    return hits / max(len(tokens), 1)


def _type_bias(entry_type: str) -> float:
    return {"task": 0.2, "macro": 0.1, "research": 0.05}.get(entry_type, 0.0)


def _format_candidate(c: dict[str, Any]) -> dict[str, Any]:
    """Format a candidate based on its type with specific fields."""
    candidate_type = c.get("type")

    if candidate_type == "macro":
        # For macros: only return id, plugin_name, plugin_description, correct_syntax, tips, site
        return {
            "id": c.get("id"),
            "type": "macro",
            "plugin_name": c.get("plugin_name"),
            "plugin_description": c.get("plugin_description"),
            "correct_syntax": c.get("correct_syntax"),
            "tips": c.get("tips", [])[:3],  # Limit to first 3 tips
            "site": c.get("site"),
            "topic": c.get("topic", []),
            "score": c.get("score"),
            "reasons": c.get("reasons", []),
        }

    elif candidate_type == "task":
        # For tasks: return specified fields only
        return {
            "id": c.get("id"),
            "name": c.get("name"),
            "data_type": c.get("data_type"),
            "question": c.get("question"),  # Add question field
            "description": c.get("description", "")[:200] + "..."
            if len(c.get("description", "")) > 200
            else c.get("description", ""),
            "requirements": c.get("requirements", {}),
            "defaults": c.get("defaults", {}),
            "tips": c.get("tips", [])[:3],
            "summary": c.get("summary"),
            "topic": c.get("topic", []),
            "workflow_steps": c.get("workflow_steps", []),
            "score": c.get("score"),
            "reasons": c.get("reasons", []),
        }
    elif candidate_type == "research":
        # For research: return specified fields only
        return {
            "name": c.get("name"),
            "content": c.get("content"),
            "summary": c.get("summary"),
            "highlights": c.get("highlights", [])[:3],  # Limit to top 3 highlights
            "references": c.get("references", [])[:5],  # Limit to top 5 references
            "topic": c.get("topic", []),
            "score": c.get("score"),
            "reasons": c.get("reasons", []),
        }
    else:
        # Fallback for unknown types
        return {
            "id": c.get("id"),
            "type": c.get("type"),
            "name": c.get("name"),
            "description": c.get("description", ""),
            "score": c.get("score"),
            "reasons": c.get("reasons", []),
        }


def _score_entry(
    query: str, image_desc: str, entry: dict[str, Any], topic: str | None = None
) -> tuple[float, dict[str, Any]]:
    q_tokens = _tokenize(query + (" " + image_desc if image_desc else ""))
    lex = _lexical_score(q_tokens, entry)
    dtm = _data_type_match(image_desc, entry)
    semantic = 0.0  # placeholder; deterministic and side-effect free

    # Add topic matching bonus
    topic_bonus = 0.0
    if topic:
        topic_tokens = _tokenize(topic.lower())
        # Check if entry's topic list contains any topic tokens
        entry_topics = entry.get("topic", [])
        if entry_topics:
            entry_topic_tokens = set()
            for t in entry_topics:
                entry_topic_tokens.update(_tokenize(str(t).lower()))
            topic_overlap = len(set(topic_tokens) & entry_topic_tokens)
            if topic_overlap > 0:
                topic_bonus = 0.2 * (topic_overlap / len(topic_tokens))  # Max 0.2 points

    score = 0.5 * lex + 0.3 * dtm + 0.2 * semantic + topic_bonus + _type_bias(entry.get("type", ""))
    reasons = []
    if lex > 0:
        reasons.append("query_match")
    if dtm > 0:
        reasons.append("data_type_match")
    if topic_bonus > 0:
        reasons.append("topic_match")
    if entry.get("tags"):
        # lightweight tag overlap check
        q = set(q_tokens)
        if any(t in q for t in [str(x).lower() for x in entry.get("tags")]):
            reasons.append("tags_overlap")
    return score, {**entry, "score": round(score, 4), "reasons": reasons}


async def kb_retrieve(
    query: Annotated[str, "Parameter Required, Main search terms describing the analysis task"],
    question: Annotated[str, "Parameter Required, User's original question for high-weight matching"],
    image_desc: Annotated[str, "Parameter Required, Basic image properties (bit depth, channels, imaging method)"],
    filters: Annotated[
        dict[str, Any], "Parameter Required, Filter by types: task|macro|research to limit search scope"
    ],
    topic: Annotated[str, "Parameter Required, Main task/goal for enhanced topic-based matching"],
    perception_info: Annotated[
        str,
        "Parameter Required, Concise image content description extracted from imagejPerception (e.g., 'DAPI nuclei', 'phase contrast cells')",
    ],
    use_enhanced_scoring: Annotated[bool, "Use enhanced scoring with TF-IDF and perception matching"] = True,
    topk: Annotated[int, "Top-k candidates to return"] = 8,
) -> str:
    """Retrieve ranked knowledge bank entries and propose a composition."""
    _ensure_dirs()
    entries = _load_registry()

    allowed_types = None
    if filters and isinstance(filters.get("types"), list):
        allowed_types = {str(t) for t in filters.get("types")}

    scored: list[dict[str, Any]] = []

    # Build TF-IDF vectorizer if using enhanced scoring
    vectorizer, tfidf_matrix = None, None
    if use_enhanced_scoring and entries:
        vectorizer, tfidf_matrix = _build_tfidf_vectorizer(entries)

    # Combine image_desc and perception_info for richer image understanding
    combined_image_desc = image_desc
    if perception_info:
        combined_image_desc += " " + perception_info

    for idx, e in enumerate(entries):
        if allowed_types and e.get("type") not in allowed_types:
            continue

        if use_enhanced_scoring:
            # Use enhanced scoring with perception matching and TF-IDF
            score, reasons = _enhanced_query_score(
                query, combined_image_desc, e, topic, question, vectorizer, tfidf_matrix, idx
            )

            # Add type bias
            score += _type_bias(e.get("type", ""))

            if score > 0.25:  # admission threshold
                scored.append({**e, "score": round(score, 4), "reasons": reasons})
        else:
            # Use original scoring method
            s, detailed = _score_entry(query, combined_image_desc, e, topic)
            if s > 0.25:  # admission threshold per spec
                scored.append(detailed)

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Apply type-specific limits to candidates
    tasks = [c for c in scored if c.get("type") == "task"][:1]  # Max 1 task
    macros = [c for c in scored if c.get("type") == "macro"][:1]  # Max 1 macro
    research = [c for c in scored if c.get("type") == "research"][:1]  # Max 1 research

    # Combine limited results
    candidates = tasks + macros + research
    # Re-sort by score after combining
    candidates.sort(key=lambda x: x["score"], reverse=True)
    # Apply overall k limit
    candidates = candidates[: max(topk, 1)]

    # Build composition: up to 1 task + up to 1 macro + up to 1 research
    comp_task = next((c for c in candidates if c.get("type") == "task"), None)
    comp_macros = [c for c in candidates if c.get("type") == "macro"][:1]
    comp_research = [c for c in candidates if c.get("type") == "research"][:1]

    confidence = 0.0
    if comp_task:
        confidence = min(1.0, float(comp_task.get("score", 0.0)))
    elif candidates:
        confidence = float(candidates[0].get("score", 0.0))

    result = {
        "candidates": [_format_candidate(c) for c in candidates],
        "composition": {
            "task": comp_task.get("id") if comp_task else None,
            "macros": [m.get("id") for m in comp_macros],
            "research": [r.get("id") for r in comp_research],
            "confidence": round(confidence, 3),
        },
        "telemetry": {
            "latency_ms": 0,
            "query_terms": _tokenize(query),
            "total_entries": len(entries),
            "scored_entries": len(scored),
        },
    }
    return json.dumps(result, ensure_ascii=False)


def _parse_steps(steps: Any) -> list[dict[str, Any]]:
    if steps is None:
        return []
    elif isinstance(steps, list):
        return steps
    elif not isinstance(steps, str):
        return []

    s = steps.strip()

    # Remove code block markers if present
    if s.startswith("```json") and s.endswith("```"):
        s = s[7:-3].strip()
    elif s.startswith("```") and s.endswith("```"):
        s = s[3:-3].strip()

    # try JSON first
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return data
        # Sometimes a dict with array under key
        if isinstance(data, dict) and "steps" in data and isinstance(data["steps"], list):
            return data["steps"]
    except Exception:
        # If JSON parsing fails, try to extract JSON from multiline string
        try:
            # Look for JSON array pattern
            lines = s.split("\n")
            json_lines = []
            in_array = False
            bracket_count = 0

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("["):
                    in_array = True
                    bracket_count += stripped.count("[") - stripped.count("]")
                    json_lines.append(stripped)
                elif in_array:
                    bracket_count += stripped.count("[") - stripped.count("]")
                    json_lines.append(stripped)
                    if bracket_count <= 0:
                        break

            if json_lines:
                json_str = "\n".join(json_lines)
                data = json.loads(json_str)
                if isinstance(data, list):
                    return data
        except Exception:
            # Intentionally ignore extraction/parsing errors here and fall back to []
            pass

    # Last fallback: treat as empty if we can't parse properly
    # This prevents creating malformed TOML files
    return []


def _toml_quote(s: str) -> str:
    """Quote string for TOML using triple quotes.

    Args:
        s: String to quote.

    Returns:
        Triple-quoted string safe for TOML.
    """
    # Escape any existing triple quotes in the content
    s = s.replace('"""', r"\"\"\"")
    return f'"""{s}"""'


def _should_use_triple_quotes(s: str, field_name: str = "") -> bool:
    """Determine if a string should use triple quotes in TOML.

    Args:
        s: String to check.
        field_name: Name of the field (for specific field rules).

    Returns:
        True if triple quotes should be used.
    """
    # Always use triple quotes for script fields
    if field_name.lower() == "script":
        return True

    # Use triple quotes for long strings (50+ chars)
    if len(s) >= 50:
        return True

    # Use triple quotes for multiline strings
    if "\n" in s:
        return True

    # Use triple quotes for strings with special characters that might cause issues
    if any(char in s for char in ['"', "\\", "\r", "\t"]):
        return True

    # Use triple quotes for certain field types that commonly contain long/complex text
    if field_name.lower() in ["task", "instructions", "description", "data_type", "content", "summary"]:
        return True

    return False


def _toml_string(s: str, field_name: str = "") -> str:
    """Format string for TOML using appropriate quoting.

    Args:
        s: String to format.
        field_name: Name of the field.

    Returns:
        Appropriately quoted string for TOML.
    """
    if _should_use_triple_quotes(s, field_name):
        return _toml_quote(s)
    else:
        # Use regular double quotes with escaping
        escaped = str(s).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'


def _canonicalize_code(code: str) -> str:
    """Canonicalize code for consistent comparison.

    Args:
        code: Raw code string.

    Returns:
        Normalized code with consistent line endings and spacing.
    """
    # Normalize line endings, trim leading and trailing spaces, collapse multiple blank lines
    s = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in s.split("\n")]
    # collapse consecutive blanks
    out: list[str] = []
    blank = False
    for ln in lines:
        if ln == "":
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    return "\n".join(out).strip()


def _normalize_url(u: str) -> str:
    try:
        p = urlparse(u.strip())
        # lower scheme and netloc
        scheme = (p.scheme or "http").lower()
        netloc = (p.netloc or "").lower()
        # remove utm_* parameters
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
        query = urlencode(q)
        return urlunparse((scheme, netloc, p.path or "", p.params or "", query, p.fragment or ""))
    except Exception:
        return u.strip()


async def kb_add_knowledge(
    summary: Annotated[str, "Plain-text summary of the dialog/workflow"],
    steps: Annotated[Any, "Minimal steps as List, JSON string, or text"],
    question: Annotated[str | None, "Optional question from user"] = None,
    topic: Annotated[list[str] | None, "Optional topic list"] = None,
    tags: Annotated[list[str] | None, "Optional tags list"] = None,
    checklist_type: Annotated[str, "task|macro|research"] = "task",
    storage: Annotated[dict[str, Any] | None, "Storage options: folder, format"] = None,
    upsert: Annotated[bool, "Whether to overwrite if exists"] = True,
    extras: Annotated[dict[str, Any] | None, "Optional type-specific extras to write"] = None,
    workflow_summary: Annotated[str | None, "Original workflow summary from kb_build"] = None,
) -> str:
    """Ingest a dialog as a checklist entry and refresh the index."""
    _ensure_dirs()

    folder = KB_ROOT / (storage.get("folder") if storage and storage.get("folder") else checklist_type)
    # If storage.folder is absolute or includes KB_ROOT, normalize to KB_ROOT-based path
    folder = Path(str(folder).replace("knowledge_bank/", "")).resolve()
    if not str(folder).startswith(str(KB_ROOT.resolve())):
        # constrain into KB_ROOT
        folder = KB_ROOT / checklist_type
    folder.mkdir(parents=True, exist_ok=True)

    # derive signature
    base = None
    if tags and len(tags):
        # Use the first tag directly if it follows our naming convention
        tag = tags[0].strip().lower()
        if "_" in tag and len(tag.split("_")) <= 3 and re.match(r"^[a-z0-9_]+$", tag):
            # Already follows our 2-case format (action_object or action_object_method)
            base = tag
        else:
            base = _slugify(tag)
    elif question:
        base = _slugify(question)
    else:
        # Use first 5 words of summary as fallback
        base = _slugify(" ".join(summary.split()[:5]))

    if not base.endswith(".v1") and ".v" not in base:
        base = f"{base}.v1"

    signature = base
    if not upsert:
        # Generate unique signature manually for non-upsert mode
        i = 2
        while (folder / f"{signature}.toml").exists():
            root = base.rsplit(".v", 1)[0] if ".v" in base else base
            signature = f"{root}.v{i}"
            i += 1

    # Build TOML content
    steps_list = _parse_steps(steps)
    lines: list[str] = []
    lines.append(f'type = "{checklist_type}"')
    lines.append(f'signature = "{signature}"')
    # Generate a meaningful default name from signature
    default_name = signature.replace("-", " ").replace(".", " ").replace("_", " ").title().replace(" V1", "")
    lines.append(f'name = "{default_name}"')
    if question:
        lines.append(f"question = {_toml_string(question, 'question')}")
    if topic:
        topic_str = ",".join(f'"{t}"' for t in topic)
        lines.append(f"topic = [{topic_str}]")
    if tags:
        tag_str = ",".join(f'"{t}"' for t in tags)
        lines.append(f"tags = [{tag_str}]")
    lines.append("")
    lines.append(f"description = {_toml_string(summary, 'description')}")
    lines.append("")

    # Add workflow summary if provided (from kb_build input)
    if workflow_summary:
        lines.append(f"summary = {_toml_string(workflow_summary, 'summary')}")
        lines.append("")

    if steps_list:
        sid = 1
        for act in steps_list:
            # Handle simple string steps
            if isinstance(act, str):
                name = act
                args = {}
            else:
                # support either {name, args} or {action: {name, args}}
                action = act.get("action") if isinstance(act, dict) and "action" in act else act
                if not isinstance(action, dict):
                    continue
                name = action.get("name") or action.get("tool")
                args = action.get("args") or {}

            if not name:
                continue
            lines.append("[[steps]]")
            lines.append(f"id = {sid}")
            lines.append("  [steps.action]")
            lines.append(f'  name = "{name}"')

            # Write all fields in action except 'name'
            for k, v in action.items():
                # Skip name and args since name is already written, args is handled separately
                if k in ("name", "args"):
                    continue
                if isinstance(v, bool):
                    lines.append(f"  {k} = {str(v).lower()}")
                elif isinstance(v, (int, float)):
                    lines.append(f"  {k} = {v}")
                elif k == "script":
                    # Special handling for script: use triple quotes and format with newlines
                    formatted_script = str(v).replace(";", ";\n")
                    lines.append(f"  {k} = {_toml_string(formatted_script, 'script')}")
                else:
                    # Use smart string formatting based on content and field name
                    lines.append(f"  {k} = {_toml_string(str(v), k)}")

            # Write args directly as flat fields if present
            if isinstance(args, dict) and args:
                for k, v in args.items():
                    if isinstance(v, bool):
                        lines.append(f"  {k} = {str(v).lower()}")
                    elif isinstance(v, (int, float)):
                        lines.append(f"  {k} = {v}")
                    elif k == "script":
                        # Special handling for script in args: use triple quotes and format with newlines
                        formatted_script = str(v).replace(";", ";\n")
                        lines.append(f"  {k} = {_toml_string(formatted_script, 'script')}")
                    else:
                        # Use smart string formatting based on content and field name
                        lines.append(f"  {k} = {_toml_string(str(v), k)}")
            lines.append("")
            sid += 1

    out_path = folder / f"{signature}.toml"
    with out_path.open("w", encoding="utf-8") as f:
        # Append type-specific extras
        if extras:
            lines.append("")

            # Task metadata
            if extras.get("task_metadata"):
                metadata = extras["task_metadata"]
                if metadata.get("name") and not metadata["name"].startswith("###"):
                    # Replace the default name with the actual name (only if it's not markdown content)
                    clean_name = metadata["name"].strip()
                    # Ensure it's a proper title, not a markdown description
                    if len(clean_name) < 100 and "\n" not in clean_name:
                        for i, line in enumerate(lines):
                            if line.startswith("name = "):
                                lines[i] = f'name = "{clean_name}"'
                                break
                if metadata.get("data_type"):
                    lines.append(f"data_type = {_toml_string(metadata['data_type'], 'data_type')}")

                # Requirements section
                if metadata.get("requirements"):
                    lines.append("[requirements]")
                    for key, value in metadata["requirements"].items():
                        if isinstance(value, list):
                            value_str = str(value).replace("'", '"')
                            lines.append(f"{key} = {value_str}")
                        else:
                            lines.append(f'{key} = "{value}"')
                    lines.append("")

                # Defaults section
                if metadata.get("defaults"):
                    lines.append("[defaults]")
                    for key, value in metadata["defaults"].items():
                        lines.append(f'{key} = "{value}"')
                    lines.append("")

                # Tips array
                if metadata.get("tips"):
                    lines.append("tips = [")
                    for tip in metadata["tips"]:
                        lines.append(f"  {_toml_string(str(tip), 'tips')},")
                    lines.append("]")
                    lines.append("")

            # [content] section for macros
            if extras.get("content") and isinstance(extras["content"], dict):
                content = extras["content"]
                lines.append("[content]")

                # Plugin name
                if content.get("plugin_name"):
                    lines.append(f'plugin_name = "{content["plugin_name"]}"')
                    lines.append("")

                # Plugin description
                if content.get("plugin_description"):
                    lines.append(
                        f"plugin_description = {_toml_string(str(content['plugin_description']), 'description')}"
                    )
                    lines.append("")

                # Correct syntax
                if content.get("correct_syntax"):
                    lines.append(f"correct_syntax = {_toml_string(str(content['correct_syntax']), 'correct_syntax')}")
                    lines.append("")

                # Tips array
                if content.get("tips") and isinstance(content["tips"], list):
                    lines.append("tips = [")
                    for tip in content["tips"]:
                        lines.append(f"  {_toml_string(str(tip), 'tips')},")
                    lines.append("]")
                    lines.append("")

                # Site
                if content.get("site"):
                    lines.append(f'site = "{content["site"]}"')
                    lines.append("")

            # Research content (only if not already handled as macro content)
            if extras.get("content") and not isinstance(extras["content"], dict):
                lines.append(f"content = {_toml_string(str(extras['content']), 'content')}")
                lines.append("")

            if extras.get("highlights"):
                highlights_str = ",".join(f'"{h}"' for h in extras["highlights"])
                lines.append(f"highlights = [{highlights_str}]")
                lines.append("")

            # references array (keep for compatibility)
            refs = extras.get("references")
            if isinstance(refs, list) and refs:
                refs_s = ",".join(f'"{str(u)}"' for u in refs)
                lines.append(f"references = [{refs_s}]")
                lines.append("")

        f.write("\n".join(lines) + "\n")

    # Refresh index
    rebuild_registry()

    return json.dumps(
        {
            "status": "ok",
            "path": str(out_path).replace("\\", "/"),
            "signature": signature,
            "notes": "Entry ingested and index refreshed.",
        },
        ensure_ascii=False,
    )


async def kb_build(
    dialog: dict,
    summary: str | None = None,
    steps: Any = None,
    *,
    question: str | None = None,
) -> str:
    _ensure_dirs()

    if not _validate_dialog(dialog):
        return _error_response("Invalid dialog parameter - must be a non-empty dictionary")

    final_summary, steps, question = _sanitize_inputs(summary, steps, question)

    _llm_prompt_template = f"""
Knowledge Base extraction action: extract reusable knowledge from the following dialog.
Dialog: {json.dumps(dialog, ensure_ascii=False)}
Dialog context summary: {final_summary}

Generate a structured JSON output for KB metadata only. Do not generate workflow JSON and do not reformat the raw steps.
The raw execution steps are passed separately and will be stored as detailed KB steps.

{{
  "task": {{
    "name": "Specific task name for filename",
    "data_type": "Detailed image/data description you can retrieved from imagej perception in dialog",
    "topic": ["topic1", "topic2"],
    "description": "Brief description of what this workflow does",
    "requirements": {{
      "bit_depth": [8, 16],
      "min_size": [128, 128],
      "channels": "requirements"
    }},
    "defaults": {{
      "key_parameter": "default_value"
    }},
    "tips": ["tip1", "tip2"]
  }},
  "macros": [{{
    "name": "macro_name_underscore_format",
    "description": "Brief macro description",
    "topic": ["topic1", "topic2"],
    "content": {{
      "plugin_name": "Exact plugin name in ImageJ",
      "plugin_description": "What the plugin does and any special requirements",
      "correct_syntax": "run(\"PluginName\", \"parameter1=value1 parameter2=value2\");",
      "tips": [
        "Error prevention tip based on dialog issues",
        "Parameter configuration guidance",
        "Common usage pitfalls to avoid"
      ],
      "site": "https://imagej.net/plugins/pluginname"
    }}
  }}],
  "research": [{{
    "name": "research_topic_underscore_format",
    "content": "Detailed research findings, analysis, and insights from the dialog. Include method comparisons, best practices, lessons learned, and technical insights.",
    "topic": ["topic1", "topic2"],
    "highlights": [
      "Key finding or insight from the research",
      "Important lesson learned or best practice",
      "Method comparison or trade-off finding"
    ],
    "references": [
      "https://relevant-paper-or-documentation.com",
      "https://github.com/relevant-repository",
      "https://imagej.net/relevant-plugin-page"
    ]
  }}]
}}

Focus on:
- Task: Extract reusable task metadata, requirements, defaults, and tips. Do not invent steps.
- Macros: Only include macros that had errors/issues(e.g. Command error, Timeout) in dialog, with prevention tips, if no error, return empty list
- Research: Capture findings and useful references from dialog, if no findings, return empty list. Only works if the dialog has Research Agent usage.

IMPORTANT: Task name must be one of these 2 formats:
1. "action_object" - e.g., "segment_nuclei", "measure_fluorescence"
2. "action_object_method" - e.g., "segment_nuclei_cellpose", "track_cells_trackmate"

Rules: lowercase, underscores only, 2-3 words maximum
"""

    try:
        structured_output = await _llm_extract_with_retry(_llm_prompt_template, dialog, final_summary, steps, question)

        # Process and persist entries
        results = {"status": "ok", "created": [], "skipped": [], "checklist_type": "task", "notes": []}

        # Deduplicate and persist task with enhanced metadata
        task_data = structured_output["task"]
        task_extras = {
            "task_metadata": {
                "name": task_data.get("name"),
                "data_type": task_data.get("data_type"),
                "question": task_data.get("question"),  # Add question field
                "requirements": task_data.get("requirements", {}),
                "defaults": task_data.get("defaults", {}),
                "tips": task_data.get("tips", []),
            }
        }

        existing_tasks = _load_existing_tasks()
        if not _is_duplicate_task(task_data, existing_tasks):
            # Use the LLM-generated task name for filename if available
            filename_tags = []
            if task_data.get("name") and task_data["name"] != "Generated workflow":
                # Convert the task name to filename format and use as primary tag
                filename_tag = task_data["name"].lower().replace(" ", "_").replace("-", "_")
                filename_tag = re.sub(r"[^a-z0-9_]", "", filename_tag)  # Clean non-alphanumeric except underscore
                if filename_tag:
                    filename_tags = [filename_tag]

            # Combine with existing tags
            final_tags = filename_tags

            task_result = await kb_add_knowledge(
                summary=task_data["description"],
                steps=task_data.get("steps", []),
                question=task_data.get("question"),
                topic=task_data.get("topic", []),
                tags=final_tags,
                checklist_type="task",
                storage=None,
                upsert=True,
                extras=task_extras,
                workflow_summary=final_summary,  # Pass the original summary from kb_build
            )
            task_result_data = json.loads(task_result)
            if task_result_data["status"] == "ok":
                results["created"].append(
                    {"type": "task", "path": task_result_data["path"], "signature": task_result_data["signature"]}
                )
        else:
            results["skipped"].append(
                {
                    "type": "task",
                    "reason": "duplicate_content",
                    "name": task_data.get("name", "Unknown"),
                    "description_preview": task_data.get("description", "")[:100] + "..."
                    if len(task_data.get("description", "")) > 100
                    else task_data.get("description", ""),
                }
            )

        # Deduplicate and persist macros with enhanced format
        existing_macros = _load_existing_macros()
        for macro in structured_output["macros"]:
            # Extract content structure
            content = macro.get("content", {})
            correct_syntax = content.get("correct_syntax", "")

            if not _is_duplicate_macro(correct_syntax, existing_macros):
                macro_extras = {
                    "content": content,  # Store the entire content structure
                }

                macro_result = await kb_add_knowledge(
                    summary=macro["description"],
                    steps=[],
                    question=macro.get("question"),
                    topic=macro.get("topic", []),
                    tags=macro.get("tags", []),
                    checklist_type="macro",
                    storage=None,
                    upsert=True,
                    extras=macro_extras,
                )
                macro_result_data = json.loads(macro_result)
                if macro_result_data["status"] == "ok":
                    results["created"].append(
                        {
                            "type": "macro",
                            "path": macro_result_data["path"],
                            "signature": macro_result_data["signature"],
                        }
                    )
            else:
                results["skipped"].append({"type": "macro", "reason": "duplicate_code", "name": macro["name"]})

        # Deduplicate and persist research with enhanced format
        existing_research = _load_existing_research()
        for research in structured_output["research"]:
            # Check against all references in the research entry
            references = research.get("references", [])
            is_duplicate = any(_is_duplicate_research(ref, existing_research) for ref in references)

            if not is_duplicate:
                results["skipped"].append(
                    {"type": "research", "reason": "duplicate_url", "urls": research.get("references", [])}
                )
                continue

            research_extras = {
                "content": research.get("content", ""),
                "references": references,
                "highlights": research.get("highlights", []),
            }

            research_result = await kb_add_knowledge(
                summary=research.get("content", ""),  # Use content as main description
                steps=[],
                question=research.get("question"),
                topic=research.get("topic", []),
                tags=research.get("tags", []),
                checklist_type="research",
                storage=None,
                upsert=True,
                extras=research_extras,
                workflow_summary=final_summary,  # Pass the original summary
            )
            research_result_data = json.loads(research_result)
            if research_result_data["status"] == "ok":
                results["created"].append(
                    {
                        "type": "research",
                        "path": research_result_data["path"],
                        "signature": research_result_data["signature"],
                    }
                )

        return json.dumps(results, ensure_ascii=False)

    except Exception as e:
        return _error_response(str(e))


def _load_existing_tasks() -> list[dict[str, Any]]:
    """Load all existing task entries for deduplication."""
    tasks = []
    for typ, path in _iter_kb_files():
        if typ != "task":
            continue

        try:
            data = _read_toml(path)
            tasks.append(
                {
                    "path": str(path),
                    "description": (data.get("description", "") or "").strip(),
                    "name": data.get("name", ""),
                    "signature": data.get("signature", ""),
                    "data_type": data.get("data_type", ""),
                    "topic": data.get("topic", []),
                    "tags": data.get("tags", []),
                }
            )
        except Exception:
            continue
    return tasks


def _load_existing_macros() -> list[dict[str, Any]]:
    """Load all existing macro entries for deduplication."""
    macros = []
    for typ, path in _iter_kb_files():
        if typ != "macro":
            continue

        try:
            data = _read_toml(path)
            content = data.get("content", {})
            # Use correct_syntax instead of code for deduplication
            correct_syntax = content.get("correct_syntax", "")
            if correct_syntax:
                macros.append(
                    {
                        "path": str(path),
                        "code": _canonicalize_code(str(correct_syntax)),
                        "name": data.get("name", ""),
                        "signature": data.get("signature", ""),
                    }
                )
        except Exception:
            continue

    return macros


def _load_existing_research() -> list[dict[str, Any]]:
    """Load all existing research entries for deduplication."""
    research = []
    for typ, path in _iter_kb_files():
        if typ != "research":
            continue

        try:
            data = _read_toml(path)
            refs = data.get("references", [])
            for ref in refs:
                research.append(
                    {
                        "path": str(path),
                        "url": _normalize_url(str(ref)),
                        "title": data.get("name", ""),
                        "signature": data.get("signature", ""),
                    }
                )
        except Exception:
            continue

    return research


def _is_duplicate_task(task_data: dict[str, Any], existing_tasks: list[dict[str, Any]]) -> bool:
    """Check if task already exists based on description similarity and topic overlap."""
    description = (task_data.get("description", "") or "").strip().lower()
    topic = set(str(t).lower() for t in task_data.get("topic", []))
    data_type = (task_data.get("data_type", "") or "").lower()

    if not description or len(description) <= 20:
        return False

    for existing in existing_tasks:
        existing_desc = existing["description"].lower()
        existing_topic = set(str(t).lower() for t in existing.get("topic", []))
        existing_data_type = existing.get("data_type", "").lower()

        if len(existing_desc) <= 20:
            continue

        desc_words = set(description.split())
        existing_words = set(existing_desc.split())
        common_words = desc_words & existing_words

        desc_similarity = False
        if len(common_words) > 0:
            similarity = len(common_words) / min(len(desc_words), len(existing_words))
            desc_similarity = similarity > 0.6

        topic_overlap = len(topic & existing_topic) > 0 if topic and existing_topic else False

        data_type_match = (
            data_type == existing_data_type if data_type and existing_data_type and data_type != "unknown" else False
        )

        if (desc_similarity and topic_overlap) or (data_type_match and (desc_similarity or topic_overlap)):
            return True

    return False


def _is_duplicate_macro(code: str, existing_macros: list[dict[str, Any]]) -> bool:
    """Check if macro code already exists (exact match after canonicalization)."""
    canonical_code = _canonicalize_code(code)
    return any(macro["code"] == canonical_code for macro in existing_macros)


def _is_duplicate_research(url: str, existing_research: list[dict[str, Any]]) -> bool:
    """Check if research URL already exists (after normalization)."""
    normalized_url = _normalize_url(url)
    return any(research["url"] == normalized_url for research in existing_research)


def _validate_dialog(dialog: dict) -> bool:
    """Validate that dialog parameter is properly formatted."""
    if not isinstance(dialog, dict):
        return False
    # Basic validation - dialog should have some content
    return bool(dialog)


def _sanitize_inputs(summary: str | None, steps: Any, question: str | None) -> tuple[str, Any, str | None]:
    """Sanitize and validate input parameters."""
    # Clean summary
    clean_summary = (summary or "").strip()
    if not clean_summary:
        clean_summary = "Generated from dialog"

    # Clean question
    clean_question = (question or "").strip() if question else None

    return clean_summary, steps, clean_question


def _error_response(error_message: str) -> str:
    """Create standardized error response."""
    return json.dumps(
        {
            "status": "error",
            "error": error_message,
            "created": [],
            "skipped": [],
            "checklist_type": "task",
        },
        ensure_ascii=False,
    )


async def _llm_extract_with_retry(
    prompt: str,
    dialog: dict,
    final_summary: str,
    steps: Any,
    question: str | None,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Extract knowledge using LLM with retry mechanism."""

    last_error = None

    for attempt in range(max_retries):
        try:
            model_client = new_model_client(load_config())
            llm_response = await model_client.create([TextMessage(role="user", text=prompt)])

            if not llm_response.content:
                last_error = f"Empty response from LLM (attempt {attempt + 1})"
                continue

            # Clean the response - remove markdown code blocks if present
            content = llm_response.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]  # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove closing ```
            content = content.strip()

            llm_data = json.loads(content)

            # Validate required structure
            if "task" not in llm_data:
                last_error = f"Invalid LLM response: missing 'task' field (attempt {attempt + 1})"
                continue

            # Return successful parsed data
            return {
                "task": {
                    "name": llm_data["task"].get("name", final_summary or "Generated workflow"),
                    "data_type": llm_data["task"].get("data_type", "unknown"),
                    "topic": llm_data["task"].get("topic", []),  # Use LLM topic if available
                    "question": question,  # Add user question as separate field
                    "description": llm_data["task"].get("description", final_summary or "Task generated from dialog"),
                    "requirements": llm_data["task"].get("requirements", {}),
                    "defaults": llm_data["task"].get("defaults", {}),
                    "tips": llm_data["task"].get("tips", []),
                    "steps": _parse_steps(steps),
                },
                "macros": llm_data.get("macros", []),
                "research": llm_data.get("research", []),
            }

        except json.JSONDecodeError as e:
            last_error = f"JSON decode error (attempt {attempt + 1}): {e}"
        except Exception as e:
            error_msg = str(e)
            if "Connection error" in error_msg or "timeout" in error_msg.lower():
                last_error = f"Network connection error (attempt {attempt + 1}): {e}"
            else:
                last_error = f"LLM call error (attempt {attempt + 1}): {e}"

    # All retries failed, raise the last error
    raise Exception(f"LLM extraction failed after {max_retries} attempts. Last error: {last_error}")
