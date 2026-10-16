"""Save the reproduction runs' 33,320 validation indices (seeded random_split over the 3.4M-doc subset)."""
from pathlib import Path

import numpy as np
import torch

from harness import ckpt_path, validation_split

OUT = Path(__file__).resolve().parent / "val_indices_repro_33320.npy"

hparams = torch.load(ckpt_path("s2"), map_location="cpu", weights_only=False)["hyper_parameters"]
val = validation_split(dict(hparams))
idx = np.asarray(val.indices, dtype=np.int64)
assert len(idx) == 33320, len(idx)
np.save(OUT, idx)
print(f"wrote {OUT}: {len(idx)} indices, max {idx.max()}")
