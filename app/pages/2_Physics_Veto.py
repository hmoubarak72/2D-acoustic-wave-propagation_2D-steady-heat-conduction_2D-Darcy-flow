import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import (  # noqa: I001
    MODE_LABELS,
    VERDICT_COLOR,
    VERDICT_ICON,
    case_cfg,
    gauge,
    heat,
    honesty_note,
    make_input,
    p0,
    run_all,
    setup,
    surrogates,
    velocity_fig,
    veto_engine,
)

from physicsveto.config import load_veto
from physicsveto.verify.checks import relative_l2

setup("Physics Veto", "🛡️")
st.title("🛡️ Physics Veto")
st.markdown(
    "Every surrogate prediction is checked against the physics it must obey, **without** using the "
    "reference solution, which in production does not exist. Then we reveal the true error so you can "
    "judge whether the veto was right."
)

case, cfg = case_cfg()
veto = veto_engine()
if veto is None or not surrogates():
    st.error("Missing weights or calibration. Run `make train bench`.")
    st.stop()
veto_cfg = load_veto()

ATTACKS = {
    "salt_body": "Insert a salt body the model never saw",
    "high_contrast": "Push velocity contrast far beyond training",
    "rough_geology": "Make the geology rough and small-scale",
    "moved_source": "Move the shot to the edge or much deeper",
    "mild_shift": "Shift the geology slightly (faster, rougher): a subtle case",
    "checkerboard_artifact": "Inject a checkerboard artifact into the prediction (simulated model bug)",
    "amplitude_drift": "Scale the predicted amplitudes by ~1.5× (simulated calibration bug)",
}

if "attack" not in st.session_state:
    st.session_state.attack = None
    st.session_state.veto_seed = 11

c_a, c_b, c_c = st.columns([1, 1, 2])
if c_a.button("✅ Normal case", width="stretch"):
    st.session_state.attack = None
    st.session_state.veto_seed = int(np.random.randint(10_000))
if c_b.button("💥 Try to break it", type="primary", width="stretch"):
    st.session_state.attack = str(np.random.choice(list(ATTACKS)))
    st.session_state.veto_seed = int(np.random.randint(10_000))
with c_c:
    pick = st.selectbox("…or choose the attack", ["(random)"] + list(ATTACKS),
                        format_func=lambda k: ATTACKS.get(k, k), label_visibility="collapsed")
    if pick != "(random)" and st.button("Run this attack"):
        st.session_state.attack = pick
        st.session_state.veto_seed = int(np.random.randint(10_000))

attack = st.session_state.attack
seed = st.session_state.veto_seed
input_mode = attack if attack in MODE_LABELS else "in_distribution"
v, src = make_input(input_mode, seed)
out = run_all(v, src)
pred, alt, ref = out["fno"].copy(), out.get("unet"), out["ref"]
rng = np.random.default_rng(seed)
if attack == "checkerboard_artifact":
    n = pred.shape[-1]
    cb = np.indices((n, n)).sum(axis=0) % 2 * 2.0 - 1.0
    pred[0::3] += rng.uniform(0.04, 0.10) * np.abs(pred[1::3][:, None]).max() * cb[None]
elif attack == "amplitude_drift":
    pred *= rng.uniform(1.35, 1.6)

rep = veto.evaluate(v, src, pred, alt)
true_err = relative_l2(pred, ref)
tol = veto_cfg["error_tolerance"]
truly_bad = true_err > tol

st.markdown(f"**Scenario:** {ATTACKS[attack] if attack else 'In-distribution input, untouched FNO prediction'}")

left, mid, right = st.columns([1.1, 1, 1.2])
with left:
    color = VERDICT_COLOR[rep.verdict]
    st.markdown(
        f'<div class="pv-verdict" style="background:{color}22;border:1px solid {color};color:{color}">'
        f"{VERDICT_ICON[rep.verdict]} {rep.verdict}</div>",
        unsafe_allow_html=True,
    )
    action = {"ACCEPT": "Use the surrogate result.",
              "REVIEW": "Borderline. An engineer should look at the evidence below.",
              "VETO": "Do not trust this prediction. Route it to the full solver."}[rep.verdict]
    st.markdown(action)
    for r in rep.reasons:
        st.markdown(f"- {r}")
with mid:
    st.plotly_chart(gauge(rep.trust_score, "Trust Score", VERDICT_COLOR[rep.verdict]), width="stretch")
with right:
    st.markdown("##### Was the veto right? (ground truth reveal)")
    st.markdown(f"True error against the reference solver: **{100 * true_err:.1f}%** "
                f"(tolerance {100 * tol:.0f}%)")
    caught = rep.verdict != "ACCEPT"
    if truly_bad and caught:
        st.success("Correct: a bad prediction was stopped.")
    elif truly_bad and not caught:
        st.error("Missed: a bad prediction got through. See the blind spots in the Benchmark Lab.")
    elif not truly_bad and rep.verdict == "VETO":
        st.warning("Conservative: an acceptable prediction was sent to the solver (costs compute, not correctness).")
    elif not truly_bad and rep.verdict == "REVIEW":
        st.info("Flagged for review, but the prediction was within tolerance.")
    else:
        st.success("Correct: a good prediction was accepted.")

st.markdown("#### The evidence")
names = [k for k in veto_cfg["checks"] if k in rep.ratios]
labels = [veto_cfg["checks"][k]["label"] for k in names]
ratios = [rep.ratios[k] for k in names]
hard = [veto_cfg["checks"][k]["hard_ratio"] for k in names]
bar_colors = ["#76b900" if r <= 1 else ("#e5484d" if r >= h else "#f2a900") for r, h in zip(ratios, hard, strict=True)]
fig = go.Figure(go.Bar(x=[min(r, 12) for r in ratios], y=labels, orientation="h", marker_color=bar_colors,
                       text=[f"{r:.2f}×" for r in ratios], textposition="outside"))
fig.add_vline(x=1.0, line_dash="dot", line_color="#9aa4b2", annotation_text="typical", annotation_position="top")
for i, h in enumerate(hard):
    fig.add_shape(type="line", x0=h, x1=h, y0=i - 0.4, y1=i + 0.4, line={"color": "#e5484d", "width": 2})
fig.update_layout(template="plotly_dark", height=300, xaxis_title="metric ÷ in-distribution level (red tick = hard veto)",
                  margin={"l": 10, "r": 40, "t": 30, "b": 10}, paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, width="stretch")
with st.expander("What each check means"):
    for k in names:
        spec = veto_cfg["checks"][k]
        st.markdown(f"**{spec['label']}** (weight {spec['weight']:.2f}, raw value {rep.metrics[k]:.4g}): {spec['explain']}")
    st.markdown("A ratio of 1× means 'as good as the worst 5% of normal, in-distribution predictions'. "
                "Thresholds live in `config/veto.yaml`.")

c1, c2, c3 = st.columns(3)
c1.plotly_chart(velocity_fig(v, src, height=320), width="stretch")
rm = rep.residual_map
c2.plotly_chart(heat(rm, "PDE residual per cell", colorscale="Inferno", zmin=0,
                     zmax=float(np.nanpercentile(rm, 99)), height=320), width="stretch")
times_ms = [a * cfg.dt * 1e3 for a in case["anchors"]]
from physicsveto.verify.checks import wave_energy  # noqa: E402

e_ref = wave_energy(ref, v, cfg)
fig_e = go.Figure()
fig_e.add_trace(go.Scatter(x=times_ms, y=rep.energy_curve / e_ref.max(), name="prediction", line={"color": "#76b900"}))
fig_e.add_trace(go.Scatter(x=times_ms, y=e_ref / e_ref.max(), name="reference (hidden from veto)",
                           line={"color": "#9aa4b2", "dash": "dot"}))
fig_e.add_vrect(x0=0, x1=(cfg.t0 + 0.75 / cfg.f0) * 1e3, fillcolor="#333", opacity=0.3, line_width=0,
                annotation_text="source active", annotation_position="top left")
fig_e.update_layout(template="plotly_dark", height=320, title={"text": "Wave energy in the domain", "font": {"size": 14}},
                    xaxis_title="time (ms)", margin={"l": 10, "r": 10, "t": 36, "b": 10},
                    paper_bgcolor="rgba(0,0,0,0)", legend={"orientation": "h", "y": -0.25})
c3.plotly_chart(fig_e, width="stretch")

k = 4
lim = float(np.abs(p0(ref)[k]).max())
d1, d2 = st.columns(2)
d1.plotly_chart(heat(p0(pred)[k], f"Prediction checked by the veto ({times_ms[k]:.0f} ms)", zmid=0, zmin=-lim, zmax=lim),
                width="stretch")
d2.plotly_chart(heat(p0(ref)[k], "Reference solver (hidden from the veto)", zmid=0, zmin=-lim, zmax=lim),
                width="stretch")
honesty_note("The veto never sees the reference. The reference is shown only to grade the veto.")
