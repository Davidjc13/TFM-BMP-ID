import random

import numpy as np

from bmpid.seed import set_global_seed


def test_seed_reproducible():
    set_global_seed(42)
    a = random.random()
    b = float(np.random.rand())
    set_global_seed(42)
    assert random.random() == a
    assert float(np.random.rand()) == b
