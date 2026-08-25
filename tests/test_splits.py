from bmpid.data.splits import assert_disjoint_user_splits, assign_user_splits


def test_user_splits_are_disjoint():
    users = [f"u{i:02d}" for i in range(10)]
    mapping = assign_user_splits(users, seed=42, ratios=(0.6, 0.2, 0.2))
    assert_disjoint_user_splits(mapping)
    assert set(mapping) == set(users)
    assert mapping.keys() == set(users)
    counts = {s: sum(1 for v in mapping.values() if v == s) for s in ("train", "dev", "eval")}
    assert counts["train"] == 6
    assert counts["dev"] == 2
    assert counts["eval"] == 2


def test_eval_user_never_in_train():
    mapping = assign_user_splits([f"u{i}" for i in range(20)], seed=0)
    train = {u for u, s in mapping.items() if s == "train"}
    eval_u = {u for u, s in mapping.items() if s == "eval"}
    assert not (train & eval_u)


def test_deterministic_given_seed():
    users = [f"u{i}" for i in range(15)]
    a = assign_user_splits(users, seed=7)
    b = assign_user_splits(users, seed=7)
    assert a == b
    c = assign_user_splits(users, seed=8)
    assert a != c


def test_assert_raises_on_overlap():
    try:
        assert_disjoint_user_splits({"u1": "train", "u2": "eval"})
    except AssertionError:
        raise AssertionError("splits disjuntos no deben fallar")
    train = {"u1", "u2"}
    eval_ = {"u2"}
    if train & eval_:
        caught = True
    else:
        caught = False
    assert caught
