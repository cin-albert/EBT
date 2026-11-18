import pytest

from decay_arms import DECAY_ARMS
from momentum import step_size

FIRST_THREE = {  # §0.2: alpha_0, alpha_1, alpha_2
    "gd_a2.5": (2.5, 2.5, 2.5), "gd_a3.5": (3.5, 3.5, 3.5), "gd_a5_harm": (5, 2.5, 5 / 3),
    "gd_a5_g0.5": (5, 2.5, 1.25), "gd_a5_g0.7": (5, 3.5, 2.45), "gd_a7.5_harm": (7.5, 3.75, 2.5),
    "gd_a7.5_g0.7": (7.5, 5.25, 3.675), "ema_m0.5_harm": (5, 2.5, 5 / 3),
}


def test_registry():
    assert list(DECAY_ARMS) == list(FIRST_THREE)
    for name, alphas in FIRST_THREE.items():
        arm = DECAY_ARMS[name]
        assert arm.name == name
        assert [step_size(arm, t) for t in range(3)] == pytest.approx(alphas)
    assert DECAY_ARMS["ema_m0.5_harm"].rule == "ema" and DECAY_ARMS["ema_m0.5_harm"].mu == 0.5
    assert all(a.rule == "gd" for n, a in DECAY_ARMS.items() if n.startswith("gd_"))
