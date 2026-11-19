from dataclasses import replace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import honesty_note, results, setup  # noqa: I001

from physicsveto.config import load_roi_defaults
from physicsveto.roi.cost import RoiInputs, compute_roi, tornado

setup("ROI & Business Case", "💰")
st.title("💰 ROI & Business Case")
st.markdown(
    "What a surrogate plus physics veto is worth to a simulation team **after** paying for every vetoed run "
    "with a full solve. Change any assumption on the left."
)

d = load_roi_defaults()
s = results("benchmark_summary.json")
measured_vr = s["in_dist_veto_rate"] if s else d["veto_rate"]

with st.sidebar:
    st.header("Workload")
    sims = st.number_input("Simulations per design cycle", 10, 1_000_000, int(d["sims_per_cycle"]), step=100)
    cycles = st.number_input("Design cycles per year", 1, 365, int(d["cycles_per_year"]))
    core_h = st.number_input("Solver core-hours per run", 1.0, 1e6, float(d["core_hours_per_run"]), step=50.0)
    usd_ch = st.number_input("$ per CPU core-hour", 0.001, 5.0, float(d["usd_per_core_hour"]), step=0.01, format="%.3f")
    wall = st.number_input("Solver wall-clock hours per run", 0.1, 1000.0, float(d["wall_hours_per_run"]))
    conc = st.number_input("Runs in parallel on the cluster", 1, 100_000, int(d["concurrent_runs"]))
    st.header("Surrogate")
    speed = st.slider("Surrogate speedup at production scale (assumption)", 2, 2000, int(d["surrogate_speedup"]))
    ev = results("veto_eval.csv")
    stress_vr = float((ev[ev["set"] != "reference_solver_output"]["verdict"] == "VETO").mean()) if ev is not None else None
    options = [f"Planning assumption ({100 * d['veto_rate']:.0f}%)",
               f"Measured, normal work only ({100 * measured_vr:.1f}%)"]
    if stress_vr is not None:
        options.append(f"Measured, stress-test mix ({100 * stress_vr:.1f}%)")
    options.append("Custom")
    vr_choice = st.radio("Veto rate", options)
    if vr_choice.startswith("Custom"):
        vr = st.slider("Custom veto rate", 0.0, 1.0, 0.15, 0.01)
    elif vr_choice.startswith("Measured, normal"):
        vr = measured_vr
    elif vr_choice.startswith("Measured, stress"):
        vr = stress_vr
    else:
        vr = d["veto_rate"]
    st.caption("Normal work in this demo is perfectly in-distribution, so its veto rate is a best case. "
               "Real workloads include unfamiliar cases; the stress-test mix is a worst case.")
    st.header("One-off costs")
    tr_sims = st.number_input("Solver runs to build training data", 0, 1_000_000, int(d["training_sims"]), step=100)
    gpu_h = st.number_input("GPU hours to train", 0.0, 1e6, float(d["gpu_training_hours"]))
    usd_gpu = st.number_input("$ per GPU hour", 0.0, 100.0, float(d["usd_per_gpu_hour"]))
    eng = st.number_input("Engineer $ per hour", 0.0, 1000.0, float(d["engineer_usd_per_hour"]))
    setup_h = st.number_input("Integration effort (engineer hours)", 0.0, 10_000.0, float(d["engineer_setup_hours"]))
    wait = st.slider("Share of solver wall time an engineer is blocked", 0.0, 1.0, float(d["engineer_wait_share"]))

p = RoiInputs(sims_per_cycle=sims, cycles_per_year=cycles, core_hours_per_run=core_h, usd_per_core_hour=usd_ch,
              surrogate_speedup=speed, veto_rate=vr, wall_hours_per_run=wall, concurrent_runs=conc,
              training_sims=tr_sims, gpu_training_hours=gpu_h, usd_per_gpu_hour=usd_gpu,
              engineer_usd_per_hour=eng, engineer_setup_hours=setup_h, engineer_wait_share=wait)
r = compute_roi(p)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Annual savings", f"${r['annual_savings_usd']:,.0f}", "compute + engineer time", delta_color="off")
c2.metric("First-year net", f"${r['first_year_net_usd']:,.0f}", f"after ${r['upfront_usd']:,.0f} one-off", delta_color="off")
be = r["break_even_months"]
c3.metric("Break-even", "never" if be == float("inf") else f"{be:.1f} months", delta_color="off")
c4.metric("Time per design cycle", f"{r['cycle_hours_hybrid']:.1f} h", f"from {r['cycle_hours_baseline']:.1f} h "
          f"(−{100 * r['time_to_design_reduction']:.0f}%)", delta_color="off")

st.markdown(
    f"**Why the veto rate dominates:** with a {speed}× surrogate and a {100 * vr:.1f}% veto rate, the effective "
    f"speedup of the whole workflow is **{r['effective_speedup']:.1f}×**, because every vetoed case still needs "
    "a full solve. Cutting the veto rate is worth more than a faster model."
)

g1, g2 = st.columns(2)
with g1:
    fig = go.Figure(go.Bar(
        x=["Solver only", "Surrogate + veto"],
        y=[r["baseline_compute_usd"], r["hybrid_compute_usd"]],
        marker_color=["#5b6472", "#76b900"],
        text=[f"${r['baseline_compute_usd']:,.0f}", f"${r['hybrid_compute_usd']:,.0f}"], textposition="outside"))
    fig.update_layout(template="plotly_dark", height=340, title="Annual compute spend",
                      margin={"l": 10, "r": 10, "t": 40, "b": 10}, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
with g2:
    rows = tornado(p)
    base = rows[0]["base"]
    fig = go.Figure()
    for row in reversed(rows):
        lo, hi = sorted([row["at_low"], row["at_high"]])
        fig.add_trace(go.Bar(y=[row["factor"]], x=[hi - lo], base=[lo], orientation="h",
                             marker_color="#76b900", showlegend=False,
                             hovertemplate=f"{row['factor']}<br>${lo:,.0f} … ${hi:,.0f}<extra></extra>"))
    fig.add_vline(x=base, line_dash="dot", line_color="#9aa4b2")
    fig.update_layout(template="plotly_dark", height=340, title="Sensitivity of first-year net (tornado)",
                      xaxis_title="first-year net ($)", margin={"l": 10, "r": 10, "t": 40, "b": 10},
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

st.markdown("#### Every assumption")
labels = {
    "sims_per_cycle": "Simulations per design cycle", "cycles_per_year": "Design cycles per year",
    "core_hours_per_run": "Solver core-hours per run", "usd_per_core_hour": "$ per CPU core-hour",
    "surrogate_speedup": "Surrogate speedup (production assumption)", "veto_rate": "Veto rate",
    "wall_hours_per_run": "Solver wall-clock hours per run", "concurrent_runs": "Runs in parallel",
    "training_sims": "Solver runs for training data", "gpu_training_hours": "GPU hours to train",
    "usd_per_gpu_hour": "$ per GPU hour", "engineer_usd_per_hour": "Engineer $ per hour",
    "engineer_setup_hours": "Integration effort (hours)", "engineer_wait_share": "Engineer blocked share",
}
src_of = {k: "default (config/roi_defaults.yaml)" for k in labels}
src_of["veto_rate"] = ("measured (results/)" if vr_choice.startswith("Measured") else
                       "custom" if vr_choice.startswith("Custom") else "planning assumption (config/roi_defaults.yaml)")
for k in labels:
    if getattr(p, k) != getattr(replace(RoiInputs(), **{kk: d[kk] for kk in labels}), k) and k != "veto_rate":
        src_of[k] = "edited"
st.dataframe(pd.DataFrame([{"Assumption": labels[k], "Value": getattr(p, k), "Source": src_of[k]} for k in labels]),
             hide_index=True, width="stretch")
honesty_note("Defaults are illustrative customer numbers, not benchmarks. The speedup is an assumption for "
             "production-scale 3D on GPU: this demo measures far smaller speedups at 64×64 on CPU "
             "(see Benchmark Lab). Veto rates labelled 'measured' come from the planted-failure benchmark.")
