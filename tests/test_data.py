import numpy as np

from physicsveto.data.fields import (
    PLANTED_INPUTS,
    encode_triplets,
    grf,
    sample_in_distribution,
    source_map,
)


def test_grf_is_standardised():
    z = grf(64, 6.0, np.random.default_rng(0))
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9


def test_in_distribution_velocity_bounds(case):
    rng = np.random.default_rng(1)
    for _ in range(20):
        v, (sz, sx) = sample_in_distribution(case, rng)
        assert v.min() >= case["v_min"] - 1e-3 and v.max() <= case["v_max"] + 1e-3
        assert sz == case["source_depth"]
        assert case["source_x_range"][0] <= sx <= case["source_x_range"][1]


def test_planted_modes_leave_training_distribution(case):
    rng = np.random.default_rng(2)
    v, _ = PLANTED_INPUTS["salt_body"](case, rng)
    assert v.max() >= 4500
    v, _ = PLANTED_INPUTS["high_contrast"](case, rng)
    assert v.max() - v.min() > case["v_max"] - case["v_min"]


def test_encode_triplets():
    pm, p0, pp = np.full((4, 4), 1.0), np.full((4, 4), 3.0), np.full((4, 4), 7.0)
    out = encode_triplets(np.stack([pm, p0, pp]))
    assert np.allclose(out[0], 3.0)
    assert np.allclose(out[1], 3.0)   # (7 - 1) / 2
    assert np.allclose(out[2], 2.0)   # 7 - 6 + 1


def test_source_map_peak():
    m = source_map(64, (16, 40))
    assert np.unravel_index(np.argmax(m), m.shape) == (16, 40)
    assert m.max() == 1.0
