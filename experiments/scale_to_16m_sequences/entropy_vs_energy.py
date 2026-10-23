"""Is the energy only an entropy score when ranking candidates of one token? Offline, from Q2 σ=3 arrays.

Usage: python entropy_vs_energy.py Q2_RUN_DIR [Q2_RUN_DIR ...]
"""
import sys
from pathlib import Path

import numpy as np

MIN_DCE = 0.1  # same pair filter as the Q2 consistency metric


def main():
    for d in map(Path, sys.argv[1:]):
        a = np.load(d / "arrays.npz")
        e, h, lp = a["e_final"], a["entropy"], a["logp"]  # (candidates, tokens)
        de, dh, dce = e[0] - e[1], h[0] - h[1], lp[1] - lp[0]  # candidate 0 vs 1
        keep = (np.abs(dce) > MIN_DCE) & (de != 0)
        right = lambda s: float((np.sign(s[keep]) == np.sign(dce[keep])).mean())  # lower score -> lower CE
        resid = de - np.polyval(np.polyfit(dh[keep], de[keep], 1), dh)  # part of ΔE not linear in ΔH
        r2 = 1 - resid[keep].var() / de[keep].var()
        bon = lambda s: float(-lp[s.argmin(0), np.arange(lp.shape[1])].mean())
        print(f"{d.name}: pairs {keep.sum()} | lower-E right {right(de):.3f} | lower-H right {right(dh):.3f} | "
              f"residual(E|H) right {right(resid):.3f} | R2(ΔE~ΔH) {r2:.3f} | BoN-{e.shape[0]} CE by E {bon(e):.4f}, "
              f"by H {bon(h):.4f} | candidate 0 {-lp[0].mean():.4f} | oracle {-lp.max(0).mean():.4f}")


if __name__ == "__main__":
    main()
