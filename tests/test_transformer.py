from pathlib import Path

import numpy as np
import pytest

from bmpid.eval.metrics import auc_score, eer_and_threshold, far_at_frr, far_frr_at_threshold, time_to_detect
from bmpid.privacy.asserts import assert_no_raw_text, forbidden_fields_present

torch = pytest.importorskip("torch")

from bmpid.losses.set2set import set2set_loss, similarity_score  # noqa: E402
from bmpid.models.behave_tf import BehaveTransformer  # noqa: E402
from bmpid.train.sampler import unpack_set2set  # noqa: E402
from bmpid.train.transformer import assert_no_imu_import, overfit_synthetic  # noqa: E402


def _batch(n: int, mods: tuple[str, ...], present: dict[str, bool] | None = None) -> dict:
    present = present or {m: True for m in mods}
    shapes = {"keystroke": (n, 50, 3), "signature": (n, 128, 6), "scroll": (n, 16, 6)}
    out = {}
    for m in mods:
        t, c = shapes[m][1], shapes[m][2]
        if present[m]:
            out[m] = torch.randn(n, t, c)
            out[f"{m}_mask"] = torch.ones(n, t, dtype=torch.bool)
            out[f"{m}_present"] = torch.ones(n, dtype=torch.bool)
        else:
            out[m] = torch.zeros(n, t, c)
            out[f"{m}_mask"] = torch.zeros(n, t, dtype=torch.bool)
            out[f"{m}_present"] = torch.zeros(n, dtype=torch.bool)
    return out


@pytest.mark.parametrize("present_n", [1, 2, 3])
def test_forward_one_two_three_modalities(present_n: int):
    mods = ("keystroke", "signature", "scroll")
    model = BehaveTransformer(modalities=mods, d_model=32, d_embed=16, nhead=4, n_layers=1, ff=64, dropout=0.0)
    flags = {m: i < present_n for i, m in enumerate(mods)}
    batch = _batch(4, mods, flags)
    e = model(batch)
    assert e.shape == (4, 16)
    assert torch.allclose(e.norm(dim=-1), torch.ones(4), atol=1e-4)


def test_imu_rejected_in_constructor():
    with pytest.raises(ValueError, match="IMU"):
        BehaveTransformer(modalities=("keystroke", "imu"))


def test_set2set_zero_when_separated():
    enroll = torch.nn.functional.normalize(torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]), dim=-1)
    pos = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    neg = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0]]), dim=-1)
    loss = set2set_loss(enroll, pos, neg, margin=0.2)
    assert float(loss) == 0.0
    sc = similarity_score(enroll, pos)
    assert float(sc) > float(similarity_score(enroll, neg))


def test_overfit_two_synthetic_users():
    cfg = {
        "seed": 0,
        "set2set_margin": 0.2,
        "model": {"d_model": 32, "d_embed": 16, "nhead": 4, "n_layers": 1, "ff": 64, "dropout": 0.0},
    }
    stats = overfit_synthetic(cfg, steps=60)
    assert stats["loss_last"] <= stats["loss_first"]
    assert stats["loss_last"] < 0.25


def test_train_transformer_source_does_not_import_imu():
    src = Path("src/bmpid/train/transformer.py").read_text(encoding="utf-8")
    assert_no_imu_import(src)
    assert "from bmpid.features.imu" not in src
    assert "import bmpid.features.imu" not in src


def test_unpack_set2set_shapes():
    e = torch.randn(14, 8)
    e = torch.nn.functional.normalize(e, dim=-1)
    enroll, pos, neg = unpack_set2set(e, batch_size=2, k_enroll=5)
    assert enroll.shape == (2, 5, 8)
    assert pos.shape == (2, 8)
    assert neg.shape == (2, 8)


def test_eval_metrics_and_no_text_in_payload():
    rng = np.random.default_rng(0)
    y = np.array([1] * 50 + [0] * 50)
    s = np.concatenate([rng.normal(1.0, 0.3, 50), rng.normal(-1.0, 0.3, 50)])
    eer, tau = eer_and_threshold(y, s)
    assert 0 <= eer < 0.2
    assert auc_score(y, s) > 0.9
    far = far_at_frr(y, s, target_frr=0.01)
    assert "far" in far
    op = far_frr_at_threshold(y, s, tau)
    assert op["far"] == pytest.approx(eer, abs=0.15)
    assert time_to_detect(np.array([1.0, -0.5, -0.6]), tau=0.0, k_consecutive=2) == 3
    payload = {"eer": eer, "auc": auc_score(y, s), "user_id": "0034"}
    assert not forbidden_fields_present(payload)
    assert_no_raw_text(payload)
    bad = {"text": "secret", "eer": 0.1}
    assert forbidden_fields_present(bad)


def test_leftover_and_min_pair_keeps_all_short_modality():
    from bmpid.eval.protocol import _query_indices

    leftover = {
        "keystroke": list(range(5, 20)),
        "signature": list(range(5, 100)),
        "scroll": list(range(5, 12)),
    }
    rng = np.random.default_rng(0)
    q = _query_indices(leftover, set(), pair="min", max_q=0, rng=rng)
    assert q["scroll"] == leftover["scroll"]
    assert len(q["keystroke"]) == len(leftover["scroll"])
    assert len(q["signature"]) == len(leftover["scroll"])
    q2 = _query_indices(leftover, {"scroll", "signature"}, pair="min", max_q=0, rng=rng)
    assert q2["keystroke"] == leftover["keystroke"]
    assert q2["scroll"] == []


def test_archive_persist_has_scores_and_no_text(tmp_path: Path):
    from bmpid.eval.archive import persist_verify

    payload = {
        "split": "eval",
        "eer": 0.2,
        "y_true": [1, 0, 1],
        "scores": [0.1, -0.2, 0.3],
        "det": {"far": [0.0], "frr": [1.0], "threshold": [1.0]},
        "per_user": [{"user_id": "0034", "n_genuine": 1}],
        "inventory": {"split": "eval", "users": []},
        "ttd_raw": {"claimed": ["0034"], "probe": ["0042"], "events": [2], "lengths": [10]},
        "probe_user_id": ["0034", "0042", "0034"],
        "claimed_user_id": ["0034", "0034", "0034"],
    }
    cfg = {"seed": 42, "privacy": {"store_raw_text": False}}
    persist_verify(tmp_path, "eval", payload, cfg)
    blob = np.load(tmp_path / "eval_scores.npz")
    assert blob["y_true"].shape == (3,)
    summary = (tmp_path / "eval_summary.json").read_text(encoding="utf-8")
    assert "0.2" in summary
    assert "secret" not in summary
    assert (tmp_path / "eval_per_user.json").is_file()
    assert (tmp_path / "eval_det.json").is_file()
    assert (tmp_path / "eval_inventory.json").is_file()
