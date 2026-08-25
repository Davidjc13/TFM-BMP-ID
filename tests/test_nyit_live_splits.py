from pathlib import Path

import pytest

from bmpid.config import REPO_ROOT, load_config
from bmpid.data.nyit import discover_users
from bmpid.data.splits import assert_disjoint_user_splits, assign_user_splits

NYIT = REPO_ROOT / "nyit_smartphone_dataset_home_lab_cleaned"


@pytest.mark.skipif(not (NYIT / "lab").is_dir(), reason="NYIT local no presente")
def test_live_nyit_user_splits_disjoint():
    cfg = load_config()
    users = discover_users(NYIT)
    assert len(users) >= 80
    ratios = tuple(cfg["splits"]["ratios"])
    mapping = assign_user_splits(users, seed=int(cfg["seed"]), ratios=ratios)
    assert_disjoint_user_splits(mapping)
    train = {u for u, s in mapping.items() if s == "train"}
    eval_u = {u for u, s in mapping.items() if s == "eval"}
    assert train and eval_u
    assert not (train & eval_u)
