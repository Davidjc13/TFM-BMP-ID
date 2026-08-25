from pathlib import Path

import numpy as np
import pytest

from bmpid.fusion.weighted import fuse_fixed, weights_inv_eer
from bmpid.privacy.asserts import assert_no_raw_text

torch = pytest.importorskip("torch")

from bmpid.losses.triplet import triplet_loss  # noqa: E402
from bmpid.models.experts import build_expert  # noqa: E402
from bmpid.train.experts import assert_no_imu_import, overfit_synthetic  # noqa: E402
from bmpid.train.sampler import unpack_set2set  # noqa: E402


def test_experts_forward_shapes():
    ks = build_expert({"model": {"d_embed": 16, "dropout": 0.0}, "train": {"ensemble": {"hidden": 8, "lstm_layers": 1}}}, "keystroke")
    batch = {
        "keystroke": torch.randn(3, 50, 3),
        "keystroke_mask": torch.ones(3, 50, dtype=torch.bool),
    }
    e = ks(batch)
    assert e.shape == (3, 16)
    assert torch.allclose(e.norm(dim=-1), torch.ones(3), atol=1e-4)

    sig = build_expert({"model": {"d_embed": 16, "dropout": 0.0}, "train": {"ensemble": {"hidden": 8, "conv_ch": 8}}}, "signature")
    sb = {"signature": torch.randn(2, 32, 6), "signature_mask": torch.ones(2, 32, dtype=torch.bool)}
    es = sig(sb)
    assert es.shape == (2, 16)

    sc = build_expert({"model": {"d_embed": 16}, "train": {"ensemble": {"scroll_hidden": [8, 8], "scroll_dropout": 0.0}}}, "scroll")
    rb = {"scroll": torch.randn(4, 1, 6), "scroll_mask": torch.ones(4, 1, dtype=torch.bool)}
    er = sc(rb)
    assert er.shape == (4, 16)


def test_triplet_zero_when_separated():
    a = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    p = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    n = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0]]), dim=-1)
    loss = triplet_loss(a, p, n, margin=0.2)
    assert float(loss) == 0.0


def test_overfit_two_synthetic_users_triplet():
    cfg = {"seed": 0, "triplet_margin": 0.2, "model": {"d_embed": 16, "dropout": 0.0}, "train": {"ensemble": {"hidden": 16, "lstm_layers": 1}}}
    stats = overfit_synthetic(cfg, steps=60)
    assert stats["loss_last"] <= stats["loss_first"]
    assert stats["loss_last"] < 0.25


def test_train_experts_source_does_not_import_imu():
    src = Path("src/bmpid/train/experts.py").read_text(encoding="utf-8")
    assert_no_imu_import(src)
    assert "from bmpid.features.imu" not in src


def test_fuse_omit_missing_renormalizes():
    scores = np.array([[1.0, 0.0, 0.5], [0.2, 0.2, 0.2]])
    present = np.array([[True, False, True], [True, True, True]])
    w = np.array([0.5, 0.25, 0.25])
    out = fuse_fixed(scores, present, w, missing="omit")
    assert out[0] == pytest.approx((0.5 * 1.0 + 0.25 * 0.5) / 0.75)
    inv = weights_inv_eer({"keystroke": 0.1, "signature": 0.2, "scroll": 0.5})
    assert inv.sum() == pytest.approx(1.0)
    assert inv[0] > inv[1] > inv[2]
    assert_no_raw_text({"eer": 0.2, "user_id": "0034"})


def test_unpack_triplet_k1():
    e = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    enroll, pos, neg = unpack_set2set(e, batch_size=2, k_enroll=1)
    assert enroll.shape == (2, 1, 8)
    assert pos.shape == (2, 8)
