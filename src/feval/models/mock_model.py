from __future__ import annotations

from typing import Any


def tokenize(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def detokenize(tokens: list[int]) -> str:
    return "".join(chr(token) for token in tokens)


def load_mock_adapter(adapter: dict[str, Any]) -> dict[str, str]:
    learned = adapter.get("learned_answers", {})
    if not isinstance(learned, dict):
        return {}
    return {str(prompt): str(answer) for prompt, answer in learned.items()}


def generate_answer(adapter: dict[str, Any], prompt: str) -> str:
    learned = load_mock_adapter(adapter)
    if prompt in learned:
        return learned[prompt]
    fallback = adapter.get("fallback_answer", "")
    return str(fallback)


def generate_rollout(adapter: dict[str, Any], prompt: str) -> dict[str, Any]:
    answer = generate_answer(adapter, prompt)
    return {
        "answer": answer,
        "tokens": tokenize(answer),
    }


def verify_rollout(adapter: dict[str, Any], prompt: str, tokens: list[int]) -> bool:
    expected = tokenize(generate_answer(adapter, prompt))
    return expected == tokens


