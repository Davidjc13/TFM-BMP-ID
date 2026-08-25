from __future__ import annotations

import logging
import re
from typing import Any

# Claves que no deben aparecer en logs (texto, caracteres, mensajes).
_FORBIDDEN_KEYS = frozenset(
    {
        "text",
        "raw_text",
        "character",
        "characters",
        "char",
        "chars",
        "message",
        "msg",
        "key_label",
        "key_name",
        "keycode",
        "key_code",
        "typed",
        "content",
        "plaintext",
        "payload",
        "keys",
        "keystrokes_text",
    }
)

_REDACTED = "[redacted]"
_KEY_HINT = re.compile(
    r"(text|character|char|typed|plaintext|keycode|key_label)",
    re.IGNORECASE,
)


def sanitize_value(obj: Any, *, key: str | None = None) -> Any:
    if key is not None and _is_forbidden_key(key):
        return _REDACTED
    if isinstance(obj, dict):
        return {k: sanitize_value(v, key=str(k)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_value(v, key=key) for v in obj)
    if isinstance(obj, str) and key is not None and _is_forbidden_key(key):
        return _REDACTED
    return obj


def _is_forbidden_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in _FORBIDDEN_KEYS:
        return True
    return bool(_KEY_HINT.search(lowered))


def format_safe(obj: Any) -> str:
    return repr(sanitize_value(obj))


class PrivacyLogFilter(logging.Filter):
    """Impide dumps de teclas / texto en el record de logging."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, dict):
            record.args = sanitize_value(record.args)
        elif isinstance(record.args, tuple):
            record.args = tuple(sanitize_value(a) for a in record.args)
        if isinstance(record.msg, (dict, list)):
            record.msg = sanitize_value(record.msg)
        elif isinstance(record.msg, str):
            record.msg = _strip_forbidden_assignments(record.msg)
        return True


_ASSIGN = re.compile(
    r"(?P<k>text|raw_text|character|characters|char|typed|plaintext|keycode|key_label)\s*[=:]\s*(?P<q>['\"]).*?(?P=q)",
    re.IGNORECASE,
)


def _strip_forbidden_assignments(msg: str) -> str:
    return _ASSIGN.sub(lambda m: f"{m.group('k')}={_REDACTED}", msg)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("bmpid")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        handler.addFilter(PrivacyLogFilter())
        logger.addHandler(handler)
    logger.addFilter(PrivacyLogFilter())
    logger.propagate = False
    return logger
