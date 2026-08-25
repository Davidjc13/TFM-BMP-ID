"""Particiones train / dev / eval disjuntas en usuarios."""

from __future__ import annotations

import random
from collections.abc import Sequence


def assign_user_splits(
    user_ids: Sequence[str],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, str]:
    """Asigna cada usuario a exactamente un split. Determinista dado `seed`."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios deben sumar 1, got {ratios}")
    unique = sorted(set(str(u) for u in user_ids))
    rng = random.Random(int(seed))
    shuffled = unique[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_dev = int(n * ratios[1])
    if n >= 3:
        n_train = max(n_train, 1)
        n_dev = max(n_dev, 1)
        n_eval = n - n_train - n_dev
        if n_eval < 1:
            n_eval = 1
            if n_dev > 1:
                n_dev -= 1
            else:
                n_train -= 1
        leftover = n - n_train - n_dev - n_eval
        n_train += leftover
    elif n == 2:
        n_train, n_dev, n_eval = 1, 0, 1
    elif n == 1:
        n_train, n_dev, n_eval = 1, 0, 0
    else:
        return {}

    mapping: dict[str, str] = {}
    i = 0
    for _ in range(n_train):
        mapping[shuffled[i]] = "train"
        i += 1
    for _ in range(n_dev):
        mapping[shuffled[i]] = "dev"
        i += 1
    for _ in range(n_eval):
        mapping[shuffled[i]] = "eval"
        i += 1
    assert len(mapping) == n
    return mapping


def assert_disjoint_user_splits(user_to_split: dict[str, str]) -> None:
    by_split: dict[str, set[str]] = {}
    for uid, split in user_to_split.items():
        by_split.setdefault(split, set()).add(uid)
    train = by_split.get("train", set())
    dev = by_split.get("dev", set())
    eval_ = by_split.get("eval", set())
    if train & eval_:
        raise AssertionError(f"user_id compartidos train/eval: {sorted(train & eval_)}")
    if train & dev:
        raise AssertionError(f"user_id compartidos train/dev: {sorted(train & dev)}")
    if dev & eval_:
        raise AssertionError(f"user_id compartidos dev/eval: {sorted(dev & eval_)}")
