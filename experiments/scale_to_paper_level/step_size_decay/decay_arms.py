"""Step-size schedule arms (IMPLEMENTATION_PLAN.md §0.2); the update rule is ../energy_momentum/momentum.py."""
from momentum import Arm

DECAY_ARMS: dict[str, Arm] = {a.name: a for a in [
    Arm("gd_a2.5", "gd", 0.0, 2.5),
    Arm("gd_a3.5", "gd", 0.0, 3.5),
    Arm("gd_a5_harm", "gd", 0.0, 5.0, "harmonic"),
    Arm("gd_a5_g0.5", "gd", 0.0, 5.0, "geom", 0.5),
    Arm("gd_a5_g0.7", "gd", 0.0, 5.0, "geom", 0.7),
    Arm("gd_a7.5_harm", "gd", 0.0, 7.5, "harmonic"),
    Arm("gd_a7.5_g0.7", "gd", 0.0, 7.5, "geom", 0.7),
    Arm("ema_m0.5_harm", "ema", 0.5, 5.0, "harmonic"),
]}
