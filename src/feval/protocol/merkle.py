from __future__ import annotations

from typing import Any

from ..utils.crypto import sha256_hex
from ..utils.jsonutil import canonical_json_bytes


EMPTY_ROOT = sha256_hex(b"feval-empty-tree")


def leaf_hash(value: Any) -> str:
    return sha256_hex(b"leaf:" + canonical_json_bytes(value))


def parent_hash(left: str, right: str) -> str:
    return sha256_hex(b"node:" + bytes.fromhex(left) + bytes.fromhex(right))


def merkle_root(leaves: list[str]) -> str:
    if not leaves:
        return EMPTY_ROOT
    level = leaves[:]
    while len(level) > 1:
        next_level: list[str] = []
        for index in range(0, len(level), 2):
            left = level[index]
            right = level[index + 1] if index + 1 < len(level) else left
            next_level.append(parent_hash(left, right))
        level = next_level
    return level[0]


def root_for_values(values: list[Any]) -> str:
    return merkle_root([leaf_hash(value) for value in values])



