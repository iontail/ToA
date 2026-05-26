import ast
import json
import math
import os
import re

from transformers import AutoTokenizer


DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
_TOKENIZER = None
_TOKENIZER_PATH = None


def get_tokenizer(model_path=None):
    global _TOKENIZER, _TOKENIZER_PATH
    path = model_path or os.getenv("TOKENIZER_PATH") or os.getenv("MODEL_PATH") or DEFAULT_MODEL
    if _TOKENIZER is None or _TOKENIZER_PATH != path:
        _TOKENIZER = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        _TOKENIZER_PATH = path
    return _TOKENIZER


def p_data(category):
    prompts = {
        "evidence_card": """
You extract one evidence card from a document chunk.
Focus on evidence, not the final answer.
Return only JSON:
{
  "claim": "claim related to the question",
  "condition": "evidence or condition supporting the claim",
  "exception": "exception that can limit or reverse the claim, or None",
  "join_key": ["people, places, events, objects, or time clues linked to other chunks"],
  "source_pointer": "chunk index",
  "answer_impact": "support, limit, refute, or irrelevant"
}
""",
        "planner": """
You are a planner for long-context reasoning.
Build an efficient traversal path over chunks using the evidence cards.
Prefer logically connected chunks, then exceptions, then conflicting evidence.
Return only JSON:
{
  "path": [chunk indexes in traversal order],
  "rationale": "short reason for the order"
}
""",
        "execute": """
You update the answer state using the current chunk and retrieved memory.
Use only the provided evidence.
Return only JSON:
{
  "evidence": "merged evidence so far",
  "answer": "A, B, C, D, or None",
  "reason": "short reasoning"
}
""",
        "draft": """
You write a draft answer from the execution trace and evidence cards.
Use only the provided evidence.
Return only JSON:
{
  "answer": "A, B, C, D, or None",
  "explanation": "evidence-based explanation"
}
""",
        "verify": """
You verify whether the draft answer is fully supported by evidence.
Revise the answer only when evidence contradicts or weakens it.
Return only JSON:
{
  "status": "pass or revise",
  "answer": "A, B, C, D, or None",
  "explanation": "verification result"
}
""",
    }
    return prompts[category]


def parse_json(model_output):
    if isinstance(model_output, dict):
        return model_output

    text = re.sub(r"\s+", " ", str(model_output)).strip()
    match = re.search(r"({.*})", text)
    text = match.group(1) if match else text

    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
            return value if isinstance(value, dict) else {}
        except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return {}


def split_text_into_token_chunks(text, chunk_count, model_path=None):
    tokenizer = get_tokenizer(model_path)
    n_chunks = max(1, chunk_count if isinstance(chunk_count, int) else len(chunk_count))
    tokens = tokenizer.tokenize(text)
    target_size = max(1, math.ceil(len(tokens) / n_chunks))
    chunks = []

    for start in range(0, len(tokens), target_size):
        chunks.append(tokenizer.convert_tokens_to_string(tokens[start:start + target_size]))

    return chunks[:n_chunks] + [""] * max(0, n_chunks - len(chunks)), len(tokens)


def truncate_text_by_tokens(text, max_tokens, model_path=None):
    tokenizer = get_tokenizer(model_path)
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.convert_tokens_to_string(tokens[:max_tokens])


def format_cards(cards, indexes=None):
    indexes = range(len(cards)) if indexes is None else indexes
    lines = []
    for idx in indexes:
        card = cards[idx]
        lines.append(
            f"[{idx}] Claim: {card['claim']} | Condition: {card['condition']} | "
            f"Exception: {card['exception']} | Join Key: {card['join_key']} | "
            f"Source: {card['source_pointer']} | Impact: {card['answer_impact']}"
        )
    return "\n".join(lines)


def normalize_card(card, chunk_id):
    return {
        "claim": str(card.get("claim", "")),
        "condition": str(card.get("condition", "")),
        "exception": str(card.get("exception", "None")),
        "join_key": normalize_join_key(card.get("join_key", [])),
        "source_pointer": f"chunk_{chunk_id}",
        "answer_impact": normalize_impact(card.get("answer_impact", "irrelevant")),
    }


def normalize_join_key(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip() and item.strip().lower() != "none"]


def normalize_impact(value):
    impact = str(value).lower().strip()
    return impact if impact in {"support", "limit", "refute", "irrelevant"} else "irrelevant"


def normalize_path(path, chunk_count, cards):
    raw_items = path if isinstance(path, list) else re.findall(r"\d+", str(path))
    ordered = []
    for item in raw_items:
        match = re.search(r"\d+", str(item))
        idx = int(match.group()) if match else -1
        if 0 <= idx < chunk_count and idx not in ordered:
            ordered.append(idx)

    focused = [
        idx for idx, card in enumerate(cards)
        if idx not in ordered and card["answer_impact"] != "irrelevant"
    ]
    planned = ordered + focused
    return planned if planned else list(range(chunk_count))


def retrieve_card_ids(cards, current_idx, limit=4):
    current_keys = {key.lower() for key in cards[current_idx]["join_key"]}
    impact_score = {"support": 3, "refute": 3, "limit": 2, "irrelevant": 0}
    scored = []

    for idx, card in enumerate(cards):
        if idx == current_idx:
            continue
        overlap = len(current_keys & {key.lower() for key in card["join_key"]})
        score = overlap * 3 + impact_score[card["answer_impact"]]
        if score:
            scored.append((score, idx))

    return [idx for _, idx in sorted(scored, reverse=True)[:limit]]


def clean_answer(value):
    match = re.search(r"\b[A-D]\b", str(value).upper())
    return match.group(0) if match else "None"
