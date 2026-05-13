"""Map evidence and answer spans to token indices."""

import re
from transformers import AutoTokenizer


def find_token_span(
    tokenizer: AutoTokenizer,
    full_text: str,
    target_span: str,
) -> dict:
    """Find token indices corresponding to a target text span.

    Uses character-level alignment with fallback to substring matching.

    Args:
        tokenizer: HuggingFace tokenizer.
        full_text: The complete input text.
        target_span: The substring to locate.

    Returns:
        dict with keys: span_text, token_indices, char_start, char_end.
    """
    # Case-insensitive search for robustness
    lower_text = full_text.lower()
    lower_span = target_span.lower()

    char_start = lower_text.find(lower_span)
    if char_start == -1:
        # Fallback: try regex word-level match
        pattern = re.escape(target_span)
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            char_start = match.start()
        else:
            return {"span_text": target_span, "token_indices": [], "char_start": -1, "char_end": -1}

    char_end = char_start + len(target_span)

    # Tokenize with character offsets
    encoded = tokenizer(full_text, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]

    token_indices = []
    for idx, (start, end) in enumerate(offsets):
        # Token overlaps with the target character span
        if start < char_end and end > char_start:
            token_indices.append(idx)

    return {
        "span_text": target_span,
        "token_indices": token_indices,
        "char_start": char_start,
        "char_end": char_end,
    }


def find_answer_position(
    tokenizer: AutoTokenizer,
    tokens: list[str],
    answer_text: str,
) -> list[int]:
    """Find token positions that correspond to the answer text.

    Args:
        tokenizer: HuggingFace tokenizer.
        tokens: List of decoded tokens from the full sequence.
        answer_text: The gold answer string.

    Returns:
        List of token indices that form the answer.
    """
    answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
    seq_len = len(tokens)

    # Try to find the exact answer token sequence in the decoded tokens
    answer_decoded = tokenizer.decode(answer_tokens).strip()

    # Sliding window over sequence
    for window_size in range(len(answer_tokens), len(answer_tokens) + 5):
        for start in range(seq_len - window_size + 1):
            window = tokenizer.decode(
                tokenizer.encode("".join(tokens[start : start + window_size]), add_special_tokens=False)
            ).strip()
            if answer_decoded.lower() in window.lower():
                return list(range(start, start + window_size))

    return []


def get_evidence_token_positions(
    tokenizer: AutoTokenizer,
    prompt: str,
    evidence: list[str],
    gold_evidence_span: str,
) -> dict:
    """Get token positions for evidence spans in a prompt.

    Args:
        tokenizer: HuggingFace tokenizer.
        prompt: Full prompt text.
        evidence: List of evidence strings.
        gold_evidence_span: The specific gold evidence span.

    Returns:
        dict with evidence_positions and all_evidence_positions.
    """
    # Find the gold evidence span
    span_result = find_token_span(tokenizer, prompt, gold_evidence_span)

    # Find all evidence token positions (rough)
    all_evidence_positions = []
    for doc in evidence:
        doc_result = find_token_span(tokenizer, prompt, doc)
        all_evidence_positions.extend(doc_result["token_indices"])

    # Deduplicate
    all_evidence_positions = sorted(set(all_evidence_positions))

    return {
        "gold_evidence_positions": span_result["token_indices"],
        "all_evidence_positions": all_evidence_positions,
        "span_result": span_result,
    }
