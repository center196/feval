from __future__ import annotations

import json
import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def numeric_value(value: Any) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


_PLAIN_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$")
_PLAIN_FRACTION = re.compile(r"^([+-]?\d+)\s*/\s*([+-]?\d+)$")
_LATEX_FRACTION = re.compile(r"^\\frac\s*\{([+-]?\d+)\}\s*\{([+-]?\d+)\}$")
_FINAL_ANSWER_MARKERS = (
    "final answer is",
    "final answer:",
    "the answer is",
    "answer is",
    "answer:",
)
_MAX_JSON_OBJECT_STARTS = 128


def _strip_math_wrappers(value: str) -> str:
    text = value.strip()
    for prefix in ("FINAL:", "Final:", "final:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    if text.startswith("\\(") and text.endswith("\\)"):
        text = text[2:-2].strip()
    elif text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    text = re.sub(r"^\\displaystyle\s*", "", text)
    return text.rstrip(". ")


def _parse_numeric_literal(value: Any) -> Fraction | None:
    text = _strip_math_wrappers(str(value))
    fraction = _LATEX_FRACTION.fullmatch(text) or _PLAIN_FRACTION.fullmatch(text)
    if fraction:
        numerator, denominator = int(fraction.group(1)), int(fraction.group(2))
        if denominator == 0:
            return None
        return Fraction(numerator, denominator)
    if not _PLAIN_NUMBER.fullmatch(text):
        return None
    try:
        decimal = Decimal(text)
    except InvalidOperation:
        return None
    if not decimal.is_finite():
        return None
    return Fraction(decimal)


def _last_balanced_boxed(text: str) -> str | None:
    r"""Return the last balanced ``\boxed{...}`` body without interpreting it."""

    search_end = len(text)
    while True:
        start = text.rfind(r"\boxed", 0, search_end)
        if start < 0:
            return None
        brace = start + len(r"\boxed")
        while brace < len(text) and text[brace].isspace():
            brace += 1
        if brace < len(text) and text[brace] == "{":
            depth = 1
            cursor = brace + 1
            while cursor < len(text):
                if text[cursor] == "{":
                    depth += 1
                elif text[cursor] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[brace + 1 : cursor].strip()
                cursor += 1
        search_end = start


def _last_answer_tag(text: str) -> str | None:
    lowered = text.lower()
    start = lowered.rfind("<answer>")
    if start < 0:
        return None
    start += len("<answer>")
    end = lowered.find("</answer>", start)
    if end < 0:
        return None
    return text[start:end].strip()


def _last_marker_candidate(text: str) -> str | None:
    lowered = text.lower()
    matches = [
        (lowered.rfind(marker), marker)
        for marker in _FINAL_ANSWER_MARKERS
        if lowered.rfind(marker) >= 0
    ]
    if not matches:
        return None
    position, marker = max(matches, key=lambda item: item[0])
    candidate = text[position + len(marker) :].strip()
    return candidate or None


def _parse_numeric_candidate(value: Any) -> Fraction | None:
    parsed = _parse_numeric_literal(value)
    if parsed is not None:
        return parsed
    boxed = _last_balanced_boxed(str(value))
    return _parse_numeric_literal(boxed) if boxed is not None else None


def strict_numeric_fraction(value: Any) -> Fraction | None:
    r"""Extract and normalize one inert numeric final answer.

    Reviewed static forms mirror common Qwen-style outputs: a complete numeric
    response, the last balanced ``\boxed{...}``, an ``<answer>`` body, or a
    final ``Answer``/``Final answer`` marker. Only an integer, decimal, or
    fraction is accepted after extraction. No expression is evaluated and no
    symbolic parser, shell, model, or dataset-supplied pattern is invoked.
    """

    text = str(value).strip()
    if not text:
        return None
    parsed = _parse_numeric_literal(text)
    if parsed is not None:
        return parsed
    tagged = _last_answer_tag(text)
    if tagged is not None:
        parsed = _parse_numeric_candidate(tagged)
        if parsed is not None:
            return parsed
    boxed = _last_balanced_boxed(text)
    if boxed is not None:
        parsed = _parse_numeric_literal(boxed)
        if parsed is not None:
            return parsed
    marked = _last_marker_candidate(text)
    return _parse_numeric_candidate(marked) if marked is not None else None


def canonical_numeric_answer(value: Any) -> str | None:
    parsed = strict_numeric_fraction(value)
    if parsed is None:
        return None
    return str(parsed.numerator) if parsed.denominator == 1 else f"{parsed.numerator}/{parsed.denominator}"


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+", text)
    return [part.strip() for part in parts if part.strip()]


def _paragraphs(text: str) -> list[str]:
    if "***" in text:
        parts = text.split("***")
    else:
        parts = re.split(r"\n\s*\n", text)
    return [part.strip() for part in parts if part.strip()]


def _divider_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\s?\*\*\*\s?", text)
    if any(not part.strip() for part in parts[1:-1]):
        return []
    return [part.strip() for part in parts if part.strip()]


def _blank_line_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\n", text) if part.strip()]


def _last_word(text: str) -> str:
    words = _words(text)
    return words[-1].lower() if words else ""


def _count_phrase(text: str, phrase: Any) -> int:
    value = str(phrase or "").strip()
    if not value:
        return 0
    pattern = re.escape(value)
    if value[0].isalnum():
        pattern = r"(?<!\w)" + pattern
    if value[-1].isalnum():
        pattern += r"(?!\w)"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _compare_count(actual: int, expected: Any, relation: Any = None) -> bool:
    target = numeric_value(expected)
    if target is None:
        return False
    target_int = int(target)
    rel = normalize_answer(relation or "exactly")
    if rel in {"less than", "fewer than", "below"}:
        return actual < target_int
    if rel in {"at most", "no more than", "less than or equal to"}:
        return actual <= target_int
    if rel in {"at least", "no less than", "greater than or equal to"}:
        return actual >= target_int
    if rel in {"more than", "greater than", "above"}:
        return actual > target_int
    return actual == target_int


def _constraint_value(params: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = params.get(name)
        if value is not None:
            return value
    return None


def _check_constraint(answer: str, constraint: dict[str, Any]) -> bool:
    constraint_id = str(constraint.get("id", ""))
    params = constraint.get("kwargs") if isinstance(constraint.get("kwargs"), dict) else {}
    lowered = answer.lower()

    if constraint_id == "keywords:forbidden_words":
        forbidden = _constraint_value(params, "forbidden_words", "keywords", "keyword")
        return all(str(word).lower() not in lowered for word in (forbidden if isinstance(forbidden, list) else [forbidden]) if word)

    if constraint_id == "keywords:existence":
        required = _constraint_value(params, "keywords", "keyword")
        values = required if isinstance(required, list) else [required]
        return bool(values) and all(_count_phrase(answer, item) > 0 for item in values if item)

    if constraint_id == "keywords:word_once":
        return _count_phrase(answer, _constraint_value(params, "keyword", "word")) == 1

    if constraint_id == "keywords:frequency":
        count = _count_phrase(answer, _constraint_value(params, "keyword", "word"))
        return _compare_count(count, _constraint_value(params, "frequency", "N"), params.get("relation"))

    if constraint_id == "keywords:no_adjacent_consecutive":
        words = _words(answer)
        initials = [word[0].lower() for word in words if word and word[0].isascii() and word[0].isalpha()]
        return all(abs(ord(left) - ord(right)) != 1 for left, right in zip(initials, initials[1:]))

    if constraint_id == "length_constraints:number_words":
        return _compare_count(len(_words(answer)), _constraint_value(params, "N", "num_words"), params.get("relation"))

    if constraint_id == "length_constraints:number_paragraphs":
        return _compare_count(len(_paragraphs(answer)), _constraint_value(params, "num_paragraphs", "N"), params.get("relation"))

    if constraint_id == "length_constraints:number_sentences":
        return _compare_count(len(_sentences(answer)), _constraint_value(params, "num_sentences", "N"), params.get("relation"))

    if constraint_id == "punctuation:no_comma":
        return "," not in answer

    if constraint_id == "punctuation:punctuation_dot":
        return "." not in answer

    if constraint_id == "punctuation:punctuation_exclamation":
        return "!" not in answer

    if constraint_id == "last_word:last_word_answer":
        expected = _constraint_value(params, "last_word", "word")
        return bool(expected) and _last_word(answer) == normalize_answer(expected)

    if constraint_id == "last_word:last_word_sent":
        expected = normalize_answer(_constraint_value(params, "last_word", "word"))
        sentences = _sentences(answer)
        return bool(expected) and bool(sentences) and all(_last_word(sentence) == expected for sentence in sentences)

    if constraint_id == "first_word:first_word_answer":
        expected = normalize_answer(_constraint_value(params, "first_word", "word"))
        words = _words(answer)
        return bool(expected) and bool(words) and words[0].lower() == expected

    if constraint_id == "startend:end_checker":
        phrase = _constraint_value(params, "end_phrase", "phrase", "suffix")
        return bool(phrase) and answer.rstrip().endswith(str(phrase))

    if constraint_id == "startend:quotation":
        stripped = answer.strip()
        return len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"'

    if constraint_id == "first_word:first_word_sent":
        expected = _constraint_value(params, "first_word", "word")
        if not expected:
            return False
        expected_text = normalize_answer(expected)
        return all(_words(sentence) and _words(sentence)[0].lower() == expected_text for sentence in _sentences(answer))

    if constraint_id == "detectable_format:title":
        return bool(re.search(r"<<[^<>]+>>", answer))

    if constraint_id == "detectable_format:number_bullet_lists":
        count = len(re.findall(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+\S", answer))
        return _compare_count(count, _constraint_value(params, "num_bullets", "num_bullet_lists", "N"), params.get("relation"))

    if constraint_id == "detectable_format:number_highlighted_sections":
        count = len(re.findall(r"(?<!\*)\*[^*\n]+\*(?!\*)", answer))
        return _compare_count(count, _constraint_value(params, "num_highlights", "N"), params.get("relation"))

    if constraint_id == "detectable_content:number_placeholders":
        count = len(re.findall(r"\[[^\[\]]+\]", answer))
        return _compare_count(count, _constraint_value(params, "num_placeholders", "N"), params.get("relation"))

    if constraint_id == "detectable_content:postscript":
        marker = _constraint_value(params, "postscript_marker", "marker")
        return bool(marker) and bool(re.search(rf"(?:^|\n)\s*{re.escape(str(marker))}", answer))

    if constraint_id == "change_case:english_capital":
        letters = [character for character in answer if character.isascii() and character.isalpha()]
        return bool(letters) and all(character.isupper() for character in letters)

    if constraint_id == "change_case:english_lowercase":
        letters = [character for character in answer if character.isascii() and character.isalpha()]
        return bool(letters) and all(character.islower() for character in letters)

    if constraint_id == "letters:letter_counting":
        count = sum(character.isalpha() for character in answer)
        return _compare_count(count, _constraint_value(params, "N", "num_letters"), params.get("relation"))

    if constraint_id == "letters:letter_counting2":
        letter = str(_constraint_value(params, "letter") or "")
        count = answer.lower().count(letter.lower()) if len(letter) == 1 else 0
        return _compare_count(count, _constraint_value(params, "let_frequency", "N"), params.get("let_relation"))

    if constraint_id == "length_constraints:nth_paragraph_first_word":
        paragraphs = _paragraphs(answer)
        expected_count = int(_constraint_value(params, "num_paragraphs") or -1)
        nth = int(_constraint_value(params, "nth_paragraph") or -1)
        expected_word = normalize_answer(_constraint_value(params, "first_word", "word"))
        return (
            len(paragraphs) == expected_count
            and 1 <= nth <= len(paragraphs)
            and bool(_words(paragraphs[nth - 1]))
            and _words(paragraphs[nth - 1])[0].lower() == expected_word
        )

    if constraint_id == "paragraphs:paragraphs":
        return len(_divider_paragraphs(answer)) == 2

    if constraint_id == "paragraphs:paragraphs2":
        return len(_blank_line_paragraphs(answer)) == 2

    if constraint_id == "count:count_unique":
        words = [word.lower() for word in _words(answer)]
        return bool(words) and len(words) == len(set(words))

    return False


def reward_instruction_constraints(answer: Any, constraints: list[dict[str, Any]]) -> int:
    text = str(answer)
    if not constraints:
        return 0
    return int(all(_check_constraint(text, constraint) for constraint in constraints))


_BOXED_LETTER = re.compile(r"\\(?:boxed|text|mathrm)\s*\{\s*([A-Za-z])\s*\}")
# Require a non-letter after the captured character so prose such as
# "the answer is complicated" cannot be mined for a stray option letter.
_ANSWER_LETTER = re.compile(r"(?i)\banswer\s*[:\-]?\s*\(?\s*([A-Za-z])(?![A-Za-z])")
_BARE_LETTER = re.compile(r"[^A-Za-z0-9]*([A-Za-z])[^A-Za-z0-9]*")


def extract_mcqa_letter(answer: Any) -> str | None:
    """Read one option letter from a response, or nothing.

    The protocol prompt asks for a bare letter, so that form is preferred. The
    two wrappers the pinned sources teach models to emit are also accepted, and
    the last such marker wins.
    """

    text = str(answer).strip()
    if not text:
        return None
    bare = _BARE_LETTER.fullmatch(text)
    if bare:
        return bare.group(1).upper()
    for pattern in (_BOXED_LETTER, _ANSWER_LETTER):
        found = pattern.findall(text)
        if found:
            return str(found[-1]).upper()
    return None


def reward_mcqa_letter(answer: Any, expected: Any) -> int:
    letter = extract_mcqa_letter(answer)
    if letter is None:
        return 0
    values = expected if isinstance(expected, list) else [expected]
    return int(any(letter == str(value).strip().upper() for value in values))


def extract_json_output(answer: Any) -> str | None:
    """Return ``output`` from a final one-field JSON object.

    Surrounding reasoning remains inert text. JSON decoding never evaluates the
    predicted program, and whitespace inside the decoded string is preserved.
    """

    text = str(answer).rstrip()
    # A Markdown code fence is presentation syntax, not part of the JSON. Only
    # a closing fence at the end is ignored; prose after the answer is rejected.
    text = re.sub(r"(?:^|\n)[ \t]*```[ \t]*$", "", text).rstrip()
    starts = [index for index, character in enumerate(text) if character == "{"]
    if not starts:
        return None
    decoder = json.JSONDecoder()
    for start in reversed(starts[-_MAX_JSON_OBJECT_STARTS:]):
        try:
            value, end = decoder.raw_decode(text, start)
        except (ValueError, TypeError):
            continue
        if text[end:].strip():
            continue
        if (
            isinstance(value, dict)
            and set(value) == {"output"}
            and isinstance(value.get("output"), str)
        ):
            return value["output"]
    return None


def reward_json_output_exact(answer: Any, expected: Any) -> int:
    values = [str(value) for value in (expected if isinstance(expected, list) else [expected])]
    output = extract_json_output(answer)
    return int(output is not None and any(output == value for value in values))


def reward_answer(answer: Any, expected: Any, tolerance: float = 1e-6, verifier: str = "exact_or_numeric") -> int:
    expected_values = expected if isinstance(expected, list) else [expected]
    if verifier == "mcqa_letter":
        return reward_mcqa_letter(answer, expected_values)
    if verifier == "json_output_exact":
        return reward_json_output_exact(answer, expected_values)
    if verifier == "label_match":
        return int(any(normalize_answer(answer) == normalize_answer(value) for value in expected_values))
    if verifier == "strict_numeric":
        answer_number = strict_numeric_fraction(answer)
        if answer_number is None:
            return 0
        answer_normalized = canonical_numeric_answer(answer_number)
        expected_normalized = [canonical_numeric_answer(value) for value in expected_values]
        return int(any(answer_normalized == value for value in expected_normalized if value is not None))
    expected_number = next((numeric_value(value) for value in expected_values if numeric_value(value) is not None), None)
    answer_number = numeric_value(answer)
    if expected_number is not None and answer_number is not None:
        return int(math.isclose(answer_number, expected_number, rel_tol=tolerance, abs_tol=tolerance))
    return int(any(normalize_answer(answer) == normalize_answer(value) for value in expected_values))


def reward_for_row(answer: Any, row: dict[str, Any], tolerance: float = 1e-6) -> int:
    verifier = row.get("verifier", "exact_or_numeric")
    if verifier == "instruction_constraints":
        return reward_instruction_constraints(answer, row.get("constraints", []))
    return reward_answer(answer, row.get("expected"), tolerance, verifier)
