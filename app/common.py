"""Shared helpers for the Streamlit pages: theme, cached resources, input builders, plots."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physicsveto.config import RESULTS_DIR, WEIGHTS_DIR, load_case, load_veto  # noqa: E402
from physicsveto.data.fields import PLANTED_INPUTS, sample_in_distribution  # noqa: E402
from physicsveto.solvers.wave import AcousticWaveSolver, WaveConfig, anchor_steps  # noqa: E402
from physicsveto.verify.trust import PhysicsVeto  # noqa: E402

GREEN = "#76b900"
AMBER = "#f2a900"
RED = "#e5484d"
VERDICT_COLOR = {"ACCEPT": GREEN, "REVIEW": AMBER, "VETO": RED}
VERDICT_ICON = {"ACCEPT": "✅", "REVIEW": "⚠️", "VETO": "⛔"}

MODE_LABELS = {
    "in_distribution": "In-distribution geology (like training)",
    "salt_body": "Salt body (fast, sharp inclusion)",
    "high_contrast": "Extreme velocity contrast",
    "rough_geology": "Rough, small-scale geology",
    "moved_source": "Shot moved to edge or deeper",
    "mild_shift": "Mild shift: slightly faster, rougher geology",
}

CSS = f"""
<style>
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.pv-card {{ border: 1px solid #2a2f3a; border-radius: 10px; padding: 14px 16px; background: #141821; }}
.pv-verdict {{ font-size: 1.6rem; font-weight: 700; padding: 10px 16px; border-radius: 10px; }}
.pv-muted {{ color: #9aa4b2; font-size: 0.9rem; }}
.pv-pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; border: 1px solid {GREEN};
           color: {GREEN}; font-size: 0.8rem; margin-right: 6px; }}
.pv-flow {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: stretch; }}
.pv-step {{ flex: 1 1 140px; border: 1px solid #2a2f3a; border-radius: 10px; padding: 10px 12px; background: #141821; }}
.pv-step b {{ color: {GREEN}; }}
</style>
"""


def setup(title: str, icon: str) -> None:
    st.set_page_config(page_title=f"PhysicsVeto · {title}", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def honesty_note(text: str) -> None:
    st.caption(f"ℹ️ {text}")


# ---------------- cached resources ----------------

@st.cache_resource
def case_cfg():
    case = load_case("case_a")
    return case, WaveConfig.from_dict(case)


@st.cache_resource
def solver() -> AcousticWaveSolver:
    return AcousticWaveSolver(case_cfg()[1])


@st.cache_resource
def surrogates():
    import torch

    from physicsveto.models.surrogate import Surrogate

    torch.set_num_threads(1)
    out = {}
    for arch in ("fno", "unet"):
        if (WEIGHTS_DIR / f"case_a_{arch}.pt").exists():
            out[arch] = Surrogate.load(WEIGHTS_DIR, arch)
    return out


@st.cache_resource
def veto_engine() -> PhysicsVeto | None:
    path = RESULTS_DIR / "calibration.json"
    if not path.exists():
        return None
    return PhysicsVeto(case_cfg()[0], load_veto(), json.loads(path.read_text()))


@st.cache_data
def results(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text())
    import pandas as pd

    return pd.read_csv(path)


def require_results() -> dict:
    s = results("benchmark_summary.json")
    if s is None:
        st.error("No benchmark results found. Run `make data train bench` first.")
        st.stop()
    return s


# ---------------- inputs ----------------

def make_input(mode: str, seed: int, src_x: int | None = None):
    case, _ = case_cfg()
    rng = np.random.default_rng(seed)
    maker = sample_in_distribution if mode == "in_distribution" else PLANTED_INPUTS[mode]
    v, src = maker(case, rng)
    if src_x is not None and mode == "in_distribution":
        src = (src[0], int(src_x))
    return v, (int(src[0]), int(src[1]))


def run_all(v, src):
    """Run FD solver + both surrogates; returns dict with predictions and wall-clock times (ms)."""
    from physicsveto.data.fields import encode_triplets

    case, cfg = case_cfg()
    s = solver()
    t = time.perf_counter()
    frames = s.solve(v, src, save_steps=anchor_steps(case["anchors"]))
    t_fd = (time.perf_counter() - t) * 1e3
    ref = encode_triplets(frames)
    out = {"ref": ref, "t_fd": t_fd}
    for arch, sur in surrogates().items():
        sur.predict(v, np.array(src))  # warm
        t = time.perf_counter()
        out[arch] = sur.predict(v, np.array(src))
        out[f"t_{arch}"] = (time.perf_counter() - t) * 1e3
    return out


# ---------------- plots ----------------

def heat(z, title, colorscale="RdBu", zmid=None, zmin=None, zmax=None, height=300, unit=""):
    if isinstance(colorscale, str) and colorscale in ("Inferno", "Magma", "Viridis", "Hot"):
        import plotly.express as px

        seq = getattr(px.colors.sequential, colorscale, None) or ["#000000", "#e5484d", "#f2a900", "#ffffff"]
        rgb = [c if c.startswith("rgb") else "rgb({},{},{})".format(*px.colors.hex_to_rgb(c)) for c in seq]
        colorscale = [[i / (len(rgb) - 1), c] for i, c in enumerate(rgb)]
    fig = go.Figure(go.Heatmap(z=z, colorscale=colorscale, zmid=zmid, zmin=zmin, zmax=zmax,
                               colorbar={"thickness": 10, "title": unit}))
    fig.update_layout(title={"text": title, "font": {"size": 14}}, height=height, template="plotly_dark",
                      margin={"l": 10, "r": 10, "t": 36, "b": 10}, paper_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(autorange="reversed", showticklabels=False, scaleanchor="x")
    fig.update_xaxes(showticklabels=False)
    return fig


def velocity_fig(v, src, height=300):
    fig = heat(v, "Velocity model (m/s)", colorscale="Viridis", height=height, unit="m/s")
    fig.add_trace(go.Scatter(x=[src[1]], y=[src[0]], mode="markers",
                             marker={"symbol": "star", "size": 14, "color": "white"}, name="shot"))
    fig.update_layout(showlegend=False)
    return fig


def gauge(value, title, color, max_value=100.0, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": suffix},
        title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": [0, max_value], "tickvals": [0, 55, 80, 100] if max_value == 100 else None},
               "bar": {"color": color},
               "steps": [{"range": [0, 55], "color": "#2b1416"}, {"range": [55, 80], "color": "#2b2414"},
                         {"range": [80, 100], "color": "#16240f"}] if max_value == 100 else []},
    ))
    fig.update_layout(height=240, template="plotly_dark", margin={"l": 40, "r": 50, "t": 50, "b": 10},
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def p0(pred):
    return pred[0::3]


# ---------------- steady cases B (heat) and C (Darcy) ----------------

STEADY_LABELS = {
    "case_b": {
        "title": "Case B · Steady heat conduction (electronics cooling)",
        "u": "Temperature rise", "k": "Conductivity k", "f": "Heat sources q", "u_unit": "K",
        "modes": {
            "in_distribution": "In-distribution board (like training)",
            "edge_hotspot": "Hot chip right next to the cold edge",
            "extreme_contrast": "Extreme conductivity contrast (copper next to an air gap)",
            "many_sources": "Six to nine small hot chips",
            "overpowered": "Same layout, 3-5x the power",
        },
    },
    "case_c": {
        "title": "Case C · Darcy flow in porous media (reservoir, groundwater, CO2 storage)",
        "u": "Pressure", "k": "Permeability k", "f": "Injection f", "u_unit": "",
        "modes": {
            "in_distribution": "In-distribution two-facies rock (like training)",
            "high_contrast": "Permeability contrast 1:100 (shale streaks)",
            "channels": "Long thin channels (fluvial geology)",
            "well_injection": "Concentrated injection well instead of uniform injection",
            "fine_scale": "Very fine-scale heterogeneity",
        },
    },
}


@st.cache_resource
def steady_bundle(case_key: str):
    """(case, solver, {name: SteadySurrogate}, SteadyVeto | None) for Case B or C."""
    import torch

    from physicsveto.models.steady import SteadySurrogate
    from physicsveto.solvers.elliptic import EllipticSolver
    from physicsveto.verify.steady import SteadyVeto

    torch.set_num_threads(1)
    case = load_case(case_key)
    sur = {a: SteadySurrogate.load(WEIGHTS_DIR, case_key, a) for a in ("fno", "unet")
           if (WEIGHTS_DIR / f"{case_key}_{a}.pt").exists()}
    path = RESULTS_DIR / case_key / "calibration.json"
    veto = SteadyVeto(case_key, load_veto(case_key), json.loads(path.read_text())) if path.exists() else None
    return case, EllipticSolver(case["n"]), sur, veto


def steady_input(case_key: str, mode: str, seed: int):
    from physicsveto.data.steady import IN_DIST, PLANTED

    case = load_case(case_key)
    maker = IN_DIST[case_key] if mode == "in_distribution" else PLANTED[case_key][mode]
    return maker(case, np.random.default_rng(seed))
