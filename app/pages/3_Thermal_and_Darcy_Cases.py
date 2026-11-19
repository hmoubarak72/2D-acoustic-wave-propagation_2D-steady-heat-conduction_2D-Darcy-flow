import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import (  # noqa: I001
    STEADY_LABELS,
    VERDICT_COLOR,
    VERDICT_ICON,
    gauge,
    heat,
    honesty_note,
    results,
    setup,
    steady_bundle,
    steady_input,
)

from physicsveto.config import load_veto

setup("Thermal & Darcy Cases", "🔥")
st.title("🔥 Thermal & Darcy cases (B and C)")
st.markdown(
    "Two steady problems with the same equation, **−∇·(k ∇u) = f** with u = 0 on the boundary, and two very "
    "different industries: chip and board cooling (Case B) and flow through rock (Case C). The FNO predicts "
    "the field, the finite-volume solver gives the reference, and the same physics veto checks every prediction."
)

case_key = st.radio("Case", ["case_b", "case_c"], format_func=lambda k: STEADY_LABELS[k]["title"], horizontal=False)
L = STEADY_LABELS[case_key]
case, solver, sur, veto = steady_bundle(case_key)
if "fno" not in sur or veto is None:
    st.error(f"Missing weights or calibration for {case_key}. Run `make train-steady bench-steady`.")
    st.stop()
veto_cfg = load_veto(case_key)
tol = veto_cfg["error_tolerance"]

ATTACKS = {**{m: L["modes"][m] for m in L["modes"] if m != "in_distribution"},
           "checkerboard_artifact": "Inject a checkerboard artifact into the prediction (simulated model bug)",
           "amplitude_drift": "Scale the prediction by ~1.4× (simulated calibration bug)"}

sk = f"steady_{case_key}"
if sk not in st.session_state:
    st.session_state[sk] = {"attack": None, "seed": 5}
state = st.session_state[sk]
c_a, c_b, c_c = st.columns([1, 1, 2])
if c_a.button("✅ Normal case", width="stretch"):
    state.update(attack=None, seed=int(np.random.randint(10_000)))
if c_b.button("💥 Try to break it", type="primary", width="stretch"):
    state.update(attack=str(np.random.choice(list(ATTACKS))), seed=int(np.random.randint(10_000)))
with c_c:
    pick = st.selectbox("…or choose the attack", ["(random)"] + list(ATTACKS), format_func=lambda k: ATTACKS.get(k, k),
                        label_visibility="collapsed", key=f"pick_{case_key}")
    if pick != "(random)" and st.button("Run this attack", key=f"run_{case_key}"):
        state.update(attack=pick, seed=int(np.random.randint(10_000)))

attack, seed = state["attack"], state["seed"]
k, f = steady_input(case_key, attack if attack in L["modes"] else "in_distribution", seed)

t = time.perf_counter()
ref = solver.solve(k, f)
t_fd = (time.perf_counter() - t) * 1e3
sur["fno"].predict(k, f)  # warm
t = time.perf_counter()
pred = sur["fno"].predict(k, f)
t_fno = (time.perf_counter() - t) * 1e3
alt = sur["unet"].predict(k, f) if "unet" in sur else None
rng = np.random.default_rng(seed)
if attack == "checkerboard_artifact":
    n = pred.shape[-1]
    pred = pred + rng.uniform(0.03, 0.06) * np.abs(pred).max() * (np.indices((n, n)).sum(axis=0) % 2 * 2.0 - 1.0)
elif attack == "amplitude_drift":
    pred = pred * rng.uniform(1.3, 1.5)

rep = veto.evaluate(k, f, pred, alt)
true_err = float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))
truly_bad = true_err > tol

st.markdown(f"**Scenario:** {ATTACKS[attack] if attack else L['modes']['in_distribution'] + ', untouched FNO prediction'}")
left, mid, right = st.columns([1.1, 1, 1.2])
with left:
    color = VERDICT_COLOR[rep.verdict]
    st.markdown(f'<div class="pv-verdict" style="background:{color}22;border:1px solid {color};color:{color}">'
                f"{VERDICT_ICON[rep.verdict]} {rep.verdict}</div>", unsafe_allow_html=True)
    st.markdown({"ACCEPT": "Use the surrogate result.",
                 "REVIEW": "Borderline. An engineer should look at the evidence below.",
                 "VETO": "Do not trust this prediction. Route it to the full solver."}[rep.verdict])
    for r in rep.reasons:
        st.markdown(f"- {r}")
with mid:
    st.plotly_chart(gauge(rep.trust_score, "Trust Score", VERDICT_COLOR[rep.verdict]), width="stretch")
with right:
    st.markdown("##### Was the veto right? (ground truth reveal)")
    st.markdown(f"True error against the reference solver: **{100 * true_err:.1f}%** (tolerance {100 * tol:.0f}%)")
    caught = rep.verdict != "ACCEPT"
    if truly_bad and caught:
        st.success("Correct: a bad prediction was stopped.")
    elif truly_bad:
        st.error("Missed: a bad prediction got through. See the blind spots in the Benchmark Lab.")
    elif rep.verdict == "VETO":
        st.warning("Conservative: an acceptable prediction was sent to the solver (costs compute, not correctness).")
    elif rep.verdict == "REVIEW":
        st.info("Flagged for review, but the prediction was within tolerance.")
    else:
        st.success("Correct: a good prediction was accepted.")

m1, m2, m3 = st.columns(3)
m1.metric("FV solver (sparse direct)", f"{t_fd:.1f} ms", delta_color="off")
m2.metric("FNO surrogate", f"{t_fno:.1f} ms", "single case, 1 thread", delta_color="off")
m3.metric("Speedup (this run)", f"{t_fd / t_fno:.2f}×", "CPU, 64×64", delta_color="off")

lim = float(max(np.abs(ref).max(), 1e-9))
c = st.columns(4)
c[0].plotly_chart(heat(np.log(k) if case_key == "case_b" else k, L["k"] + (" (log)" if case_key == "case_b" else ""),
                       colorscale="Viridis", height=300), width="stretch")
c[1].plotly_chart(heat(f, L["f"], colorscale="Hot", zmin=0, height=300), width="stretch")
c[2].plotly_chart(heat(ref, f"{L['u']}: solver", colorscale="Inferno", zmin=0, zmax=lim, height=300), width="stretch")
c[3].plotly_chart(heat(pred, f"{L['u']}: FNO (checked by veto)", colorscale="Inferno", zmin=0, zmax=lim, height=300),
                  width="stretch")
d1, d2, d3 = st.columns([1, 1, 1.3])
err_map = np.abs(pred - ref)
d1.plotly_chart(heat(err_map, f"|FNO − solver| (max {100 * err_map.max() / lim:.1f}% of peak)", colorscale="Magma",
                     zmin=0, zmax=float(err_map.max()) + 1e-9, height=300), width="stretch")
rm = rep.residual_map
d2.plotly_chart(heat(rm, "PDE residual per cell", colorscale="Inferno", zmin=0,
                     zmax=float(np.nanpercentile(rm, 99)) + 1e-12, height=300), width="stretch")
names = [n for n in veto_cfg["checks"] if n in rep.ratios]
ratios = [rep.ratios[n] for n in names]
hard = [veto_cfg["checks"][n]["hard_ratio"] for n in names]
fig = go.Figure(go.Bar(x=[min(r, 12) for r in ratios], y=[veto_cfg["checks"][n]["label"] for n in names],
                       orientation="h", text=[f"{r:.2f}×" for r in ratios], textposition="outside",
                       marker_color=["#76b900" if r <= 1 else ("#e5484d" if r >= h else "#f2a900")
                                     for r, h in zip(ratios, hard, strict=True)]))
fig.add_vline(x=1.0, line_dash="dot", line_color="#9aa4b2")
fig.update_xaxes(range=[0, 1.3 * max(1.0, *[min(r, 12) for r in ratios])])
fig.update_layout(template="plotly_dark", height=300, title={"text": "Checks (metric ÷ in-distribution level)",
                                                             "font": {"size": 14}},
                  margin={"l": 10, "r": 40, "t": 36, "b": 10}, paper_bgcolor="rgba(0,0,0,0)")
d3.plotly_chart(fig, width="stretch")
with st.expander("What each check means"):
    for n in names:
        spec = veto_cfg["checks"][n]
        st.markdown(f"**{spec['label']}** (raw value {rep.metrics[n]:.4g}): {spec['explain']}")
    st.markdown("Thresholds live in `config/veto_steady.yaml`; levels are calibrated on validation data in "
                f"`results/{case_key}/calibration.json`.")
honesty_note("The veto never sees the reference solution; it is shown only to grade the veto. At 64×64 a sparse "
             "direct solve is a few milliseconds, so the surrogate is not faster here; the point of this page is "
             "the trust check, and the Benchmark Lab has the measured numbers.")

s = results(f"{case_key}/benchmark_summary.json")
if s:
    st.markdown("#### Measured on the held-out test set")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("FNO median error (in-dist.)", f"{100 * s['fno_in_dist_rel_l2_median']:.1f}%", delta_color="off")
    e2.metric("Veto rate on in-dist. cases", f"{100 * s['in_dist_veto_rate']:.0f}%", delta_color="off")
    e3.metric("Bad cases caught (flag recall)", f"{100 * s['flag_recall']:.0f}%", delta_color="off")
    e4.metric("VETO precision", f"{100 * s['veto_precision']:.0f}%", delta_color="off")
    honesty_note(f"From results/{case_key}/benchmark_summary.json, {s['n_eval']} evaluation cases, "
                 f"{s['hardware']}.")
