import sys
from pathlib import Path

WS = Path(__file__).resolve().parent
REPO = WS.parents[2]
for p in (REPO, REPO / "experiments/thinking_study/common", REPO / "experiments/thinking_study/q3_energy_landscape", WS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
