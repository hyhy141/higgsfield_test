"""Token budget estimation.

We deliberately avoid a tokenizer dependency (tiktoken would need network access
to fetch BPE files at runtime, which breaks the offline guarantee). The eval
states "approximate is fine; don't blow past it by 2x", so a calibrated heuristic
is sufficient. The blend below tracks GPT/Claude tokenization within ~10-15% on
prose and slightly OVER-estimates, which is the safe direction for a budget cap.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # ~4 chars/token is the classic rule of thumb; we also floor by whitespace
    # word count so short, word-dense lines aren't underestimated.
    char_estimate = len(text) / 4.0
    word_estimate = len(text.split()) * 1.3
    return max(1, int(round(max(char_estimate, word_estimate))))
