"""Config loading helpers (YAML files in config/)."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"
WEIGHTS_DIR = ROOT / "weights"
DATA_DIR = ROOT / "data"


@cache
def _load(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_case(key: str = "case_a") -> dict:
    return dict(_load("cases.yaml")[key])


def load_veto(case_key: str = "case_a") -> dict:
    return dict(_load("veto.yaml" if case_key == "case_a" else "veto_steady.yaml"))


def load_roi_defaults() -> dict:
    return dict(_load("roi_defaults.yaml"))
