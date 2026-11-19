"""Generate datasets for Cases B and C.   python scripts/generate_data_steady.py [case_b case_c]"""

from __future__ import annotations

import sys
import time

import numpy as np

from physicsveto.config import DATA_DIR, load_case
from physicsveto.data.steady import PLANTED, build_steady_set

N_PLANTED = 40


def main(keys: list[str]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for key in keys:
        case = load_case(key)
        rng = np.random.default_rng(case["seed"])
        t0 = time.time()
        for split in ("train", "val", "test"):
            k, f, u = build_steady_set(key, case, case[f"n_{split}"], rng)
            np.savez_compressed(DATA_DIR / f"{key}_{split}.npz", k=k, f=f, u=u)
        planted = {}
        for name, maker in PLANTED[key].items():
            k, f, u = build_steady_set(key, case, N_PLANTED, rng, maker=maker)
            planted[f"{name}__k"], planted[f"{name}__f"], planted[f"{name}__u"] = k, f, u
        np.savez_compressed(DATA_DIR / f"{key}_planted.npz", **planted)
        print(f"{key}: done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or ["case_b", "case_c"])
