"""SHA-256 de ficheros del corpus (reproducibilidad, no LFS)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

_CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_sha256sums(root: Path, rel_paths: Iterable[str], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for rel in sorted(set(rel_paths)):
        full = root / rel
        digest = sha256_file(full)
        lines.append(f"{digest}  {rel.replace(chr(92), '/')}\n")
    dest.write_text("".join(lines), encoding="utf-8")
    return dest


def parse_sha256sums(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        if not rel:
            digest, _, rel = line.partition(" ")
        out[rel.strip()] = digest.strip()
    return out


def verify_sha256sums(root: Path, sums_path: Path) -> list[str]:
    """Devuelve rutas que no coinciden. Lista vacía = OK."""
    expected = parse_sha256sums(sums_path)
    bad: list[str] = []
    for rel, digest in expected.items():
        full = root / rel
        if not full.is_file() or sha256_file(full) != digest:
            bad.append(rel)
    return bad
