"""Generate the Case A datasets (train / val / test / planted failures) as compressed .npz."""

from __future__ import annotations

import time

import numpy as np

from physicsveto.config import DATA_DIR, load_case
from physicsveto.data.fields import PLANTED_INPUTS, build_set

N_PLANTED = 40


def main() -> None:
    case = load_case("case_a")
    out = DATA_DIR
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(case["seed"])
    t0 = time.time()
    for split in ("train", "val", "test"):
        v, s, y = build_set(case, case[f"n_{split}"], rng)
        np.savez_compressed(out / f"case_a_{split}.npz", v=v, src=s, y=y)
        print(f"{split}: {len(v)} samples  ({time.time() - t0:.1f}s)")
    planted = {}
    for name, maker in PLANTED_INPUTS.items():
        v, s, y = build_set(case, N_PLANTED, rng, maker=maker)
        planted[f"{name}__v"], planted[f"{name}__src"], planted[f"{name}__y"] = v, s, y
    np.savez_compressed(out / "case_a_planted.npz", **planted)
    print(f"planted: {len(PLANTED_INPUTS)} modes x {N_PLANTED}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
