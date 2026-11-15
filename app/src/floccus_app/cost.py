"""Approximate screening cost with a paid API (prices are user-supplied).

calls  = LLM criteria x (records + layer-2 share x records)
tokens = (real system prompt + average record text) / chars-per-token
cost   = input tokens x input price + output tokens x output price (per million)

It is an estimate: tokenisation differs between models and answers vary in
length. Local Ollama costs nothing (price 0).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from biofloc_review.protocol import Protocol

from floccus_app.screening import system_prompt

CHARS_PER_TOKEN = 3.5
OUTPUT_TOKENS_PER_CALL = 120
USER_TEMPLATE_CHARS = 150
DEFAULT_USER_CHARS = 1700  # mean of the biofloc corpus (title + abstract + keywords + venue)
DEFAULT_LAYER2_SHARE = 0.25  # biofloc: 4,203 of 16,965 records went to layer 2


@dataclass(frozen=True)
class Estimate:
    calls: int
    input_tokens: int
    output_tokens: int
    cost: float


def average_user_chars(records: Iterable[Mapping[str, Any]]) -> int:
    lengths = []
    for r in records:
        text = sum(len(str(r.get(k) or "")) for k in ("title", "abstract", "keywords", "venue"))
        lengths.append(text)
    if not lengths:
        return DEFAULT_USER_CHARS
    return round(sum(lengths) / len(lengths) + USER_TEMPLATE_CHARS)


def estimate(p: Protocol, n_records: int, layer2_share: float, user_chars: int,
             price_in: float, price_out: float) -> Estimate:
    llm = [c for c in p.criteria if not c.deterministic]
    per_record_chars = sum(len(system_prompt(p, c)) for c in llm) + len(llm) * user_chars
    screened = n_records + round(n_records * layer2_share)
    calls = len(llm) * screened
    input_tokens = round(per_record_chars * screened / CHARS_PER_TOKEN)
    output_tokens = calls * OUTPUT_TOKENS_PER_CALL
    total = input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out
    return Estimate(calls=calls, input_tokens=input_tokens, output_tokens=output_tokens, cost=total)
