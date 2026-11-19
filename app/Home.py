import streamlit as st
from common import GREEN, honesty_note, results, setup  # noqa: I001

setup("Overview", "🛡️")

st.title("🛡️ PhysicsVeto")
st.markdown(
    "### AI surrogates make simulation much faster. Physics decides whether you can trust the answer."
)
st.markdown(
    "PhysicsVeto runs an AI surrogate for a physics solver, then checks **every prediction** against "
    "the physics it is supposed to obey. Predictions that fail are routed back to the full solver "
    "instead of being trusted. The measured veto rate feeds straight into the business case."
)

st.markdown("#### How it works")
st.markdown(
    f"""
<div class="pv-flow">
  <div class="pv-step"><b>1 · Input</b><br/>Velocity model + shot position (a seismic survey design case)</div>
  <div class="pv-step"><b>2 · Surrogate</b><br/>Fourier Neural Operator predicts the wavefield in milliseconds</div>
  <div class="pv-step"><b>3 · Physics veto</b><br/>Wave-equation residual, absorbing-boundary check, energy balance, OOD distance, FNO-vs-U-Net agreement</div>
  <div class="pv-step"><b>4 · Verdict</b><br/><span style="color:{GREEN}">✅ ACCEPT</span> · ⚠️ REVIEW · ⛔ VETO → full solver</div>
  <div class="pv-step"><b>5 · Business case</b><br/>Savings net of every vetoed run</div>
</div>
""",
    unsafe_allow_html=True,
)

s = results("benchmark_summary.json")
st.markdown("#### Headline results · Case A, 2D acoustic wave (seismic)")
if s is None:
    st.warning("No results yet. Run `make data train bench` to produce them.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FNO error, in-distribution", f"{100 * s['fno_in_dist_rel_l2_median']:.1f}%",
              "median relative L2", delta_color="off")
    c2.metric("Veto precision / recall", f"{100 * s['veto_precision']:.1f}% / {100 * s['veto_recall']:.1f}%",
              f"{s['n_eval']} predictions, {s['n_truly_bad']} truly bad", delta_color="off")
    bad_before = 100 * s["unfiltered_share_truly_bad"]
    bad_after = 100 * (s["accepted_share_truly_bad"] or 0)
    c3.metric("Bad answers that get through", f"{bad_after:.1f}%", f"vs {bad_before:.0f}% with no veto",
              delta_color="off")
    c4.metric("Veto rate on normal work", f"{100 * s['in_dist_veto_rate']:.1f}%",
              "in-distribution test set", delta_color="off")
    honesty_note(
        f"All numbers are read from results/ and produced by scripts/benchmark.py. "
        f"Truly bad = relative L2 error above {100 * s['error_tolerance']:.0f}% against the reference solver. "
        f"The evaluation mixes in-distribution cases with planted failures, so 'no veto' is a stress-test baseline."
    )
    sc = results("scaling.json")
    st.markdown("#### Honest speed statement")
    msg = (
        f"At this demo's scale (64×64, 2D, CPU) the finite-difference solver takes **{s['fd_solver_ms']:.0f} ms** "
        f"and the FNO takes **{s['fno_batched_ms_per_sample']:.1f} ms** per case when batched "
        f"(**{s['speedup_batched']:.2f}×**). "
    )
    if sc:
        last = sc["measured"][-1]
        msg += (
            f"The gap grows with resolution: at {last['n']}×{last['n']} the measured speedup is "
            f"**{last['speedup']:.1f}×** on one CPU thread (solver cost grows ~N^{sc['fd_exponent']}, "
            f"FNO ~N^{sc['fno_exponent']}). Production 3D models on GPUs are where 100-1000× is claimed; "
            "that is not measured here."
        )
    st.markdown(msg)

rows = [(k, results(f"{k}/benchmark_summary.json")) for k in ("case_b", "case_c")]
if any(r for _, r in rows):
    st.markdown("#### Same veto, two more physics · Case B (heat conduction) and Case C (Darcy flow)")
    cols = st.columns(4)
    i = 0
    for key, r in rows:
        if not r:
            continue
        name = "Heat (B)" if key == "case_b" else "Darcy (C)"
        cols[i].metric(f"{name}: FNO error", f"{100 * r['fno_in_dist_rel_l2_median']:.1f}%", "median, in-dist.",
                       delta_color="off")
        cols[i + 1].metric(f"{name}: bad cases caught", f"{100 * r['flag_recall']:.0f}%",
                           f"veto rate on normal work {100 * r['in_dist_veto_rate']:.0f}%", delta_color="off")
        i += 2
    honesty_note("From results/case_b and results/case_c (scripts/benchmark_steady.py). Bad cases caught = "
                 "share of truly bad predictions sent to VETO or REVIEW.")

st.markdown("#### Walk through the demo")
st.page_link("pages/1_Simulate_and_Compare.py", label="Simulate & Compare: solver vs surrogate", icon="🧪")
st.page_link("pages/2_Physics_Veto.py", label="Physics Veto: try to break it", icon="🛡️")
st.page_link("pages/2_Thermal_and_Darcy_Cases.py", label="Thermal & Darcy cases (B and C): same veto, new physics",
             icon="🔥")
st.page_link("pages/3_ROI_Business_Case.py", label="ROI & Business Case", icon="💰")
st.page_link("pages/4_Benchmark_Lab.py", label="Benchmark Lab", icon="📊")
st.page_link("pages/5_GTM_Studio.py", label="Go-to-Market Studio", icon="🎯")
st.page_link("pages/6_Roadmap_and_PRD.py", label="Roadmap & PRD", icon="🗺️")

st.divider()
st.caption(
    "Independent portfolio project. Not affiliated with or endorsed by NVIDIA or any vendor named in this app. "
    "2D, 64×64, synthetic data: a working prototype of the workflow, not a production solver."
)
