"""Real-task data loader for FEVER, HotpotQA, and other QA datasets.

Maps real QA formats to the controlled reasoning sample schema used by ICI.
"""

import json
from pathlib import Path


def load_fever_samples(path: str | Path, max_samples: int = 100) -> list[dict]:
    """Load FEVER-format samples and convert to ICI schema.

    FEVER format: {claim, evidence, label} where label in {SUPPORTS, REFUTES, NOT ENOUGH INFO}

    Maps to reasoning_type:
      SUPPORTS → direct_evidence
      REFUTES → conflict
      NOT ENOUGH INFO → evidence_gap
    """
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                item = json.loads(line) if line.strip() else {}
            except json.JSONDecodeError:
                continue

            if not item:
                continue

            label = item.get("label", "")
            rt_map = {"SUPPORTS": "direct_evidence", "REFUTES": "conflict",
                       "NOT ENOUGH INFO": "evidence_gap"}
            reasoning_type = rt_map.get(label, "direct_evidence")

            evidence_text = item.get("evidence", "")
            if isinstance(evidence_text, list):
                evidence_text = " ".join(evidence_text)

            sample = {
                "id": f"fever_{i:04d}",
                "source": "fever",
                "evidence": [evidence_text] if evidence_text else [],
                "question": f"Verify: {item.get('claim', '')}",
                "gold_answer": label,
                "gold_evidence_span": evidence_text[:200] if evidence_text else "",
                "reasoning_type": reasoning_type,
                "gold_thought_steps": [],
                "label": "faithful" if label == "SUPPORTS" else "unfaithful" if label == "REFUTES" else "faithful",
            }
            samples.append(sample)

    return samples


def load_hotpotqa_samples(path: str | Path, max_samples: int = 100) -> list[dict]:
    """Load HotpotQA-format samples and convert to ICI schema.

    HotpotQA format: {question, answer, supporting_facts, context}

    Maps to:
      multi_step (always, since HotpotQA is multi-hop)
      with supporting_facts providing gold evidence spans
    """
    samples = []
    with open(path) as f:
        data = json.load(f)

    for i, item in enumerate(data[:max_samples]):
        question = item.get("question", "")
        answer = item.get("answer", "")
        context = item.get("context", {})
        supporting_facts = item.get("supporting_facts", [])

        # Build evidence from supporting facts
        evidence_docs = []
        gold_spans = []
        for title, sent_idx in supporting_facts:
            if title in context and int(sent_idx) < len(context[title]):
                sent = context[title][int(sent_idx)]
                evidence_docs.append(f"{title}: {sent}")
                gold_spans.append(sent)

        # Add a few distractor sentences
        all_docs = []
        for title, sentences in context.items():
            for j, sent in enumerate(sentences):
                prefix = f"Doc {title}: " if title not in [s[0] for s in supporting_facts] else f"Doc {title}: "
                all_docs.append(f"{prefix}{sent}")

        sample = {
            "id": f"hotpot_{i:04d}",
            "source": "hotpotqa",
            "evidence": all_docs[:10],  # limit context
            "question": question,
            "gold_answer": answer,
            "gold_evidence_span": " ".join(gold_spans[:2]) if gold_spans else "",
            "reasoning_type": "multi_step",
            "gold_thought_steps": [],
            "label": "faithful",
        }
        samples.append(sample)

    return samples


def load_controlled_real_mix(
    controlled_path: str | Path,
    real_paths: dict[str, str | Path] | None = None,
    max_controlled: int = 50,
    max_real: int = 50,
) -> list[dict]:
    """Mix controlled reasoning samples with real task samples.

    Args:
        controlled_path: Path to toy_reasoning.jsonl.
        real_paths: {"fever": path, "hotpotqa": path} or None to use controlled only.
        max_controlled, max_real: Max samples from each source.

    Returns:
        Combined list of samples with 'source' field.
    """
    from src.utils import load_jsonl

    samples = load_jsonl(controlled_path)[:max_controlled]

    if real_paths:
        if "fever" in real_paths:
            fever = load_fever_samples(real_paths["fever"], max_samples=max_real)
            samples.extend(fever)
        if "hotpotqa" in real_paths:
            hotpot = load_hotpotqa_samples(real_paths["hotpotqa"], max_samples=max_real)
            samples.extend(hotpot)

    return samples
