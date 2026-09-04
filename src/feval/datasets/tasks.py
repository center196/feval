"""Per-source normalisation into one protocol-owned evaluation row.

Every source ships its own prompt template, its own answer wrapper, and its own
grading conventions. Each source row is reduced to a bare question plus a
machine-checkable expected value, and Feval then builds the prompt and picks a
reviewed local verifier with equivalent answer semantics. Source-provided code,
regular expressions, and verifier metadata are never compiled or executed.

Only three verifiers exist, and all three are pure string work:

* ``strict_numeric``     exact Fraction equality on a complete number
* ``mcqa_letter``        one option letter
* ``json_output_exact``  exact match of a predicted program output

No LLM judge, no symbolic algebra, and no code execution anywhere.
"""

from __future__ import annotations

import ast
import hashlib
import re
from typing import Any, Callable

from .rewards import canonical_numeric_answer


MATH_SUFFIX = (
    "\n\nReturn only the final answer as an integer, decimal, or fraction. "
    "Do not include reasoning."
)
MCQA_SUFFIX = (
    "\n\nAnswer with only the single letter of the correct option. "
    "Do not include reasoning."
)
CODE_OUTPUT_SUFFIX = (
    "\n\nReturn only a JSON object of the form {\"output\": \"...\"} holding the "
    "exact predicted output string. Do not include reasoning."
)

# Multiple-choice rows are kept only at this width. Narrower questions hand a
# miner a large free score from blind guessing, which is the one failure this
# protocol can least afford.
REQUIRED_MCQA_OPTIONS = 10
MCQA_LETTERS = tuple("ABCDEFGHIJ")

_OPTION_LINE = re.compile(r"(?m)^[ \t]*([A-Z])(?:[:.]|\))[ \t]+")
_NUMBERED_PART = re.compile(r"(?m)^[ \t]*\d+[.)][ \t]+")
# OpenScienceReasoning-2 stores its answer bare in some row groups and
# LaTeX-wrapped in others. Both encodings name the same option letter.
_WRAPPED_LETTER = re.compile(
    r"\\(?:text|mathrm|mathbf|mathit|boxed)\s*\{\s*([A-Za-z])\s*\}"
)
_PLAIN_LETTER = re.compile(r"\$?\s*([A-Za-z])\s*\$?\s*[.)]?")


def _unwrap_letter(value: Any) -> str | None:
    """Read the option letter out of either encoding the sources use."""

    text = str(value or "").strip()
    for pattern in (_WRAPPED_LETTER, _PLAIN_LETTER):
        match = pattern.fullmatch(text)
        if match:
            return match.group(1).upper()
    return None


# Preambles the sources bake into their prompts; the protocol supplies its own.
_STRIP_PREFIXES = (
    "Solve the following problem. Make sure to put the answer "
    "(and only answer) inside \\boxed{}.",
)
_STRIP_SUFFIXES = (
    "Return your response as a json with a field 'output' that contains the "
    "predicted output string.",
)
# ``verification_info`` arrives as a Python repr, not JSON. It is parsed into an
# AST and only a literal string node at the reviewed key is read. The AST is
# never compiled or executed, and the bound limits parser resource use.
MAX_LITERAL_BYTES = 64 * 1024


# A source prompt that still tells the model how to format its answer would
# contradict the protocol's own instruction. Such rows are dropped rather than
# patched, because a half-stripped template is worse than one fewer row.
_CONFLICTING_FORMAT = re.compile(
    r"(?i)last line of your response"
    r"|inside \\boxed"
    r"|put your (final )?answer"
    r"|return your response as a json"
    r"|<answer>|<think>"
)


def _strip_template(text: str) -> str:
    value = str(text or "").strip()
    for prefix in _STRIP_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix):].strip()
    for suffix in _STRIP_SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)].strip()
    return value


def _strip_rendered_template(content: str, template_prompt: str) -> str:
    """Drop a rendered instruction paragraph that precedes the question.

    The source renders its template with a randomised example letter and an
    option list sized to the question, so the stored template never matches the
    prompt byte for byte. Only the lead-in before the first quoted example is
    stable, and the question always begins after the first blank line.
    """

    prefix = str(template_prompt or "").split("{problem}")[0]
    head = prefix.split("'")[0].strip()
    if len(head) < 20 or "\n\n" not in content:
        return content
    lead, rest = content.split("\n\n", 1)
    return rest if lead.strip().startswith(head[:40]) else content


def _clean_question(text: str) -> str | None:
    value = _strip_template(text)
    if not value or _CONFLICTING_FORMAT.search(value):
        return None
    return value


def _content_id(text: str) -> str:
    """Stable id for a source that ships no identifier of its own.

    A counter would renumber the same question in every window, which would
    defeat cross-window de-duplication and make a row_id meaningless outside
    the window that produced it.
    """

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:20]


def _static_string_field(value: Any, field: str) -> str | None:
    """Read one string field from a dict or Python-dict representation.

    ``ast.parse`` builds inert syntax nodes. This function accepts only a
    constant string key paired with a constant string value and never calls
    ``eval``, ``literal_eval``, ``compile``, or any dataset-supplied callable.
    """

    if isinstance(value, dict):
        result = value.get(field)
        return result if isinstance(result, str) else None
    if not isinstance(value, str) or len(value) > MAX_LITERAL_BYTES:
        return None
    try:
        parsed = ast.parse(value, mode="eval")
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return None
    body = parsed.body
    if not isinstance(body, ast.Dict):
        return None
    matches: list[str] = []
    for key_node, value_node in zip(body.keys, body.values):
        if (
            isinstance(key_node, ast.Constant)
            and key_node.value == field
            and isinstance(value_node, ast.Constant)
            and isinstance(value_node.value, str)
        ):
            matches.append(value_node.value)
    return matches[0] if len(matches) == 1 else None


def _option_letters(text: str) -> set[str]:
    return {match.group(1) for match in _OPTION_LINE.finditer(text)}


def _declared_option_letters(value: Any) -> set[str] | None:
    """Read populated option keys from the source's inert ``options`` value."""

    if not isinstance(value, list) or not 1 <= len(value) <= 26:
        return None
    letters: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        populated = [
            str(key).strip().upper()
            for key, option in item.items()
            if isinstance(option, str) and option.strip()
        ]
        if len(populated) != 1:
            return None
        letter = populated[0]
        if len(letter) != 1 or letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return None
        letters.append(letter)
    if len(set(letters)) != len(letters):
        return None
    return set(letters)


def _single_answer_question(text: str) -> bool:
    """Conservatively reject prompts that visibly request multiple answers."""

    question_marks = text.count("?") + text.count("？")
    return question_marks <= 1 and len(_NUMBERED_PART.findall(text)) <= 1


def _mcqa_row(
    *,
    row_id: str,
    question: str,
    answer: str,
    source: str,
    license_id: str | None,
    declared_letters: set[str] | None = None,
) -> dict[str, Any] | None:
    letters = _option_letters(question)
    if letters != set(MCQA_LETTERS):
        return None
    if declared_letters is not None and declared_letters != letters:
        return None
    letter = _unwrap_letter(answer)
    if letter is None or letter not in letters:
        return None
    return {
        "row_id": row_id,
        "task_type": "mcqa",
        "prompt": question + MCQA_SUFFIX,
        "expected": [letter],
        "verifier": "mcqa_letter",
        "source_dataset": source,
        "license": license_id,
    }


def _math_row(
    *, row_id: str, question: str, answer: Any, source: str, license_id: str | None
) -> dict[str, Any] | None:
    expected = canonical_numeric_answer(answer)
    if not expected or not question:
        return None
    return {
        "row_id": row_id,
        "task_type": "math",
        "prompt": question + MATH_SUFFIX,
        "expected": [expected],
        "verifier": "strict_numeric",
        "source_dataset": source,
        "license": license_id,
    }


# --------------------------------------------------------------------------
# One normaliser per pinned source. Each returns None for any row the protocol
# cannot check deterministically; rejection is always the safe direction.
# --------------------------------------------------------------------------

def normalize_open_math_reasoning(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    # Only rows whose answer the upstream pipeline actually extracted, and then
    # only those whose answer is a complete number rather than an expression.
    if str(row.get("problem_type") or "") != "has_answer_extracted":
        return None
    problem = _clean_question(row.get("problem"))
    if problem is None:
        return None
    return _math_row(
        row_id=f"math:omr:{_content_id(problem)}",
        question=problem,
        answer=row.get("expected_answer"),
        source="nvidia/OpenMathReasoning",
        license_id="cc-by-4.0",
    )


def normalize_crossthink_math(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    metadata = row.get("meta_data") if isinstance(row.get("meta_data"), dict) else {}
    reward = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
    if str(reward.get("style") or "") != "rule":
        return None
    question = _clean_question(metadata.get("question"))
    if question is None or not _single_answer_question(question):
        return None
    return _math_row(
        row_id=f"math:crossthink:{metadata.get('index', index)}",
        question=question,
        answer=reward.get("ground_truth"),
        source="nvidia/Nemotron-CrossThink",
        license_id="cc-by-4.0",
    )


def normalize_knowledge_mcqa(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    params = row.get("responses_create_params")
    params = params if isinstance(params, dict) else {}
    inputs = params.get("input") if isinstance(params.get("input"), list) else []
    content = ""
    for item in inputs:
        if isinstance(item, dict) and item.get("content"):
            content = str(item["content"])
            break
    template = row.get("template_metadata") if isinstance(row.get("template_metadata"), dict) else {}
    question = _clean_question(
        _strip_rendered_template(content, str(template.get("template_prompt") or ""))
    )
    if question is None:
        return None
    declared_letters = _declared_option_letters(row.get("options"))
    if declared_letters is None:
        return None
    return _mcqa_row(
        row_id=f"mcqa:knowledge:{row.get('uuid') or index}",
        question=question,
        answer=row.get("expected_answer"),
        source="nvidia/Nemotron-RL-knowledge-mcqa",
        license_id="cc-by-4.0",
        declared_letters=declared_letters,
    )


def normalize_open_science(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    question = _clean_question(row.get("input"))
    if question is None:
        return None
    return _mcqa_row(
        row_id=f"mcqa:science:{_content_id(question)}",
        question=question,
        answer=row.get("expected_answer"),
        source="nvidia/OpenScienceReasoning-2",
        license_id="cc-by-4.0",
    )


def normalize_code_understanding(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    truth = _static_string_field(row.get("verification_info"), "ground_truth")
    # An empty expected output is guessable and carries no signal.
    if not isinstance(truth, str) or not truth:
        return None
    prompt = _clean_question(row.get("prompt"))
    if not prompt:
        return None
    return {
        "row_id": f"code_output:vcu:{row.get('problem_id') or index}",
        "task_type": "code_output",
        "prompt": prompt + CODE_OUTPUT_SUFFIX,
        "expected": [truth],
        "verifier": "json_output_exact",
        "source_dataset": "PrimeIntellect/synthetic-code-understanding",
        "license": "apache-2.0",
    }


Normalizer = Callable[[dict[str, Any], int], "dict[str, Any] | None"]

NORMALIZERS: dict[str, Normalizer] = {
    "open_math_reasoning": normalize_open_math_reasoning,
    "crossthink_math": normalize_crossthink_math,
    "knowledge_mcqa": normalize_knowledge_mcqa,
    "open_science": normalize_open_science,
    "code_understanding": normalize_code_understanding,
}
