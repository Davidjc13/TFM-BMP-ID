from __future__ import annotations

from typing import Any, Iterable

_TEXT_FIELDS = frozenset(
    {
        "text",
        "raw_text",
        "character",
        "characters",
        "char",
        "typed",
        "plaintext",
        "message",
    }
)


def forbidden_fields_present(obj: Any, *, keys: Iterable[str] | None = None) -> list[str]:
    banned = set(keys) if keys is not None else set(_TEXT_FIELDS)
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                name = str(k).lower()
                here = f"{path}.{k}" if path else str(k)
                if name in banned:
                    found.append(here)
                walk(v, here)
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(obj, "")
    return found


def assert_no_raw_text(obj: Any) -> None:
    hits = forbidden_fields_present(obj)
    if hits:
        raise AssertionError(f"Campos de texto no permitidos en persistencia: {hits}")
