import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import MODE_LABELS, STEADY_LABELS, honesty_note, require_results, results, setup  # noqa: I001

setup("Benchmark Lab", "📊")
st.title("📊 Benchmark Lab")
s = require_results()
st.markdown(f"Case A · {s['case']} · timings on {s['hardware']}. Every number below comes from `results/`.")

st.markdown("#### Leaderboard")
lb = pd.DataFrame(results("leaderboard.json"))
lb_show = lb.rename(columns={
    "model": "Model", "rel_l2_median": "Rel. L2 error (median)", "latency_ms": "Latency, 1 case (ms)",
    "batched_ms_per_sample": "Batched (ms / case)", "params": "Parameters", "train_minutes": "Training (CPU min)",
    "in_dist_veto_rate": "Veto rate (in-dist.)",
})
st.dataframe(lb_show, hide_index=True, width="stretch",
             column_config={"Rel. L2 error (median)": st.column_config.NumberColumn(format="%.3f"),
                            "Veto rate (in-dist.)": st.column_config.NumberColumn(format="%.3f")})
honesty_note("The veto runs on FNO predictions, using U-Net only as the second opinion for the disagreement check. "
             "The PINN baseline and the physics-loss FNO are measured on Case B, further down this page.")

st.markdown("#### Does the veto work? Planted-failure evaluation")
c1, c2, c3, c4 = st.columns(4)
c1.metric("VETO precision", f"{100 * s['veto_precision']:.0f}%", delta_color="off")
c2.metric("VETO recall", f"{100 * s['veto_recall']:.0f}%", delta_color="off")
c3.metric("Flag (VETO+REVIEW) recall", f"{100 * s['flag_recall']:.0f}%", delta_color="off")
c4.metric("Exact solver output accepted", f"{100 * s['reference_output_accept_rate']:.0f}%", delta_color="off")
pm = pd.DataFrame(s["per_mode"]).T.reset_index().rename(columns={"index": "set"})
pm["label"] = pm["set"].map(lambda k: MODE_LABELS.get(k, k.replace("_", " ")))
pm_show = pm[["label", "group", "n", "share_truly_bad", "veto_rate", "flag_rate", "recall_of_bad", "median_rel_l2"]]
st.dataframe(pm_show.rename(columns={
    "label": "Test set", "group": "Type", "n": "Cases", "share_truly_bad": "Truly bad",
    "veto_rate": "Vetoed", "flag_rate": "Vetoed or review", "recall_of_bad": "Bad cases caught",
    "median_rel_l2": "Median FNO error"}), hide_index=True, width="stretch")
honesty_note(f"'Truly bad' means relative L2 error above {100 * s['error_tolerance']:.0f}% against the reference "
             "solver. Physics checks cannot see every failure: a uniform amplitude error barely registers on the "
             "PDE residual and is caught by the model-disagreement check. Precision is below recall by design: "
             "a false alarm costs one solver run, a miss costs a wrong answer.")

ev = results("veto_eval.csv")
if ev is not None:
    ev = ev[ev["set"] != "reference_solver_output"].copy()
    ev["Test set"] = ev["set"].map(lambda k: MODE_LABELS.get(k, k.replace("_", " ")))
    g1, g2 = st.columns(2)
    with g1:
        fig = px.scatter(ev, x="trust_score", y="rel_l2_fno", color="verdict", hover_data=["Test set"],
                         color_discrete_map={"ACCEPT": "#76b900", "REVIEW": "#f2a900", "VETO": "#e5484d"},
                         template="plotly_dark", log_y=True, labels={"trust_score": "Trust Score",
                                                                     "rel_l2_fno": "true error (rel. L2)"})
        fig.add_hline(y=s["error_tolerance"], line_dash="dot", line_color="#9aa4b2", annotation_text="tolerance")
        fig.update_layout(height=380, title="Trust Score vs true error", paper_bgcolor="rgba(0,0,0,0)",
                          margin={"l": 10, "r": 10, "t": 40, "b": 10})
        st.plotly_chart(fig, width="stretch")
    with g2:
        fig = px.scatter(ev, x="m_ood_distance", y="rel_l2_fno", color="Test set", template="plotly_dark",
                         log_x=True, log_y=True,
                         labels={"m_ood_distance": "OOD distance (Mahalanobis)", "rel_l2_fno": "true error"})
        fig.update_layout(height=380, title="Accuracy vs distance from training data", paper_bgcolor="rgba(0,0,0,0)",
                          margin={"l": 10, "r": 10, "t": 40, "b": 10}, legend={"font": {"size": 10}})
        st.plotly_chart(fig, width="stretch")

sc = results("scaling.json")
if sc:
    st.markdown("#### Where does the surrogate start paying off?")
    df = pd.DataFrame(sc["measured"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.n, y=df.fd_ms, name="FD solver", mode="lines+markers", line={"color": "#9aa4b2"}))
    fig.add_trace(go.Scatter(x=df.n, y=df.fno_ms_per_sample, name="FNO inference", mode="lines+markers",
                             line={"color": "#76b900"}))
    fig.update_layout(template="plotly_dark", height=360, xaxis_title="grid size N (N×N)", yaxis_title="ms per case",
                      xaxis_type="log", yaxis_type="log", paper_bgcolor="rgba(0,0,0,0)",
                      margin={"l": 10, "r": 10, "t": 10, "b": 10})
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df.rename(columns={"n": "N", "cells": "Cells", "time_steps": "Time steps", "fd_ms": "FD (ms)",
                                    "fno_ms_per_sample": "FNO (ms)", "speedup": "Speedup"}).round(2),
                 hide_index=True, width="stretch")
    honesty_note(f"{sc['hardware']}. {sc['note']} Fitted exponents: FD ~N^{sc['fd_exponent']}, "
                 f"FNO ~N^{sc['fno_exponent']}. FNO timings use the production architecture with untrained "
                 "weights (cost does not depend on the weights).")

hist = {a: results(f"train_history_{a}.json") for a in ("fno", "unet")}
if all(hist.values()):
    st.markdown("#### Training curves (CPU)")
    fig = go.Figure()
    for a, color in (("fno", "#76b900"), ("unet", "#8fa3ff")):
        h = pd.DataFrame(hist[a])
        fig.add_trace(go.Scatter(x=h.epoch, y=h.val_rel_l2_median, name=a.upper(), line={"color": color}))
    fig.update_layout(template="plotly_dark", height=300, yaxis_title="validation rel. L2 (median)",
                      xaxis_title="epoch", yaxis_type="log", paper_bgcolor="rgba(0,0,0,0)",
                      margin={"l": 10, "r": 10, "t": 10, "b": 10})
    st.plotly_chart(fig, width="stretch")


# ---------------- Cases B and C ----------------
st.divider()
st.markdown("### Cases B and C · steady heat conduction and Darcy flow")
rows = []
for ck in ("case_b", "case_c"):
    ss = results(f"{ck}/benchmark_summary.json")
    if ss:
        rows.append({"Case": STEADY_LABELS[ck]["title"].split(" · ")[1], "FNO error (median)": ss["fno_in_dist_rel_l2_median"],
                     "U-Net error (median)": ss["unet_in_dist_rel_l2_median"], "Veto rate in-dist.": ss["in_dist_veto_rate"],
                     "VETO precision": ss["veto_precision"], "VETO recall": ss["veto_recall"],
                     "Flag recall": ss["flag_recall"], "Bad answers accepted": ss["accepted_share_truly_bad"],
                     "Bad with no veto": ss["unfiltered_share_truly_bad"], "Solver (ms)": ss["fd_solver_ms"],
                     "FNO batched (ms)": ss["fno_batched_ms_per_sample"], "Eval cases": ss["n_eval"]})
if not rows:
    st.info("No Case B/C results yet. Run `make data-steady train-steady pinn bench-steady`.")
else:
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in rows[0]
                                if c not in ("Case", "Eval cases", "Solver (ms)", "FNO batched (ms)")})
    honesty_note("Same veto design and scoring as Case A, with the energy check replaced by a conservation "
                 "(boundary-flux balance) check. At 64×64 a sparse direct solve takes a few milliseconds, so the "
                 "surrogate is not faster than the solver here; that only changes at larger, 3D or repeated-query scale.")
    ck = st.radio("Details for", ["case_b", "case_c"], horizontal=True,
                  format_func=lambda k: STEADY_LABELS[k]["title"].split(" · ")[0])
    ss = results(f"{ck}/benchmark_summary.json")
    if ss:
        modes = STEADY_LABELS[ck]["modes"]
        pm = pd.DataFrame(ss["per_mode"]).T.reset_index().rename(columns={"index": "set"})
        pm["Test set"] = pm["set"].map(lambda k: modes.get(k, k.replace("_", " ")))
        st.dataframe(pm[["Test set", "group", "n", "share_truly_bad", "veto_rate", "flag_rate", "recall_of_bad",
                         "median_rel_l2"]].rename(columns={
            "group": "Type", "n": "Cases", "share_truly_bad": "Truly bad", "veto_rate": "Vetoed",
            "flag_rate": "Vetoed or review", "recall_of_bad": "Bad cases caught", "median_rel_l2": "Median FNO error"}),
            hide_index=True, width="stretch")
    lbs = results(f"{ck}/leaderboard.json")
    if lbs:
        st.markdown("#### Leaderboard")
        st.dataframe(pd.DataFrame(lbs).rename(columns={
            "model": "Model", "rel_l2_median": "Rel. L2 error (median)", "latency_ms": "Latency, 1 case (ms)",
            "batched_ms_per_sample": "Batched (ms / case)", "params": "Parameters", "train_minutes": "Training (CPU min)",
            "n_train": "Training samples", "physics_weight": "Physics-loss weight",
            "residual_median": "PDE residual (median)"}), hide_index=True, width="stretch")

    st.markdown("#### Data-only vs physics-informed vs PINN (Case B)")
    lb_b = {r["model"]: r for r in (results("case_b/leaderboard.json") or [])}
    pinn = results("case_b/pinn.json")
    cmp_rows = [{"Approach": name, "Training samples": r["n_train"], "Median error": r["rel_l2_median"],
                 "PDE residual (median)": r["residual_median"], "Training (CPU min)": r["train_minutes"],
                 "New instance costs": "one forward pass"}
                for name, r in lb_b.items() if r["n_train"] > 0]
    if pinn:
        cmp_rows.append({"Approach": "Classic PINN (no data, trained per instance)", "Training samples": 0,
                         "Median error": pinn["median_rel_l2"], "PDE residual (median)": None,
                         "Training (CPU min)": round(pinn["median_train_seconds"] / 60, 2),
                         "New instance costs": f"a new {pinn['median_train_seconds']:.0f} s training run"})
    if cmp_rows:
        st.dataframe(pd.DataFrame(cmp_rows), hide_index=True, width="stretch",
                     column_config={"Median error": st.column_config.NumberColumn(format="%.3f"),
                                    "PDE residual (median)": st.column_config.NumberColumn(format="%.3f")})
        honesty_note("Physics loss = the same discrete PDE residual the veto checks, added to the training loss "
                     "(PINO-style). The PINN uses no solver data but must be retrained for every new board layout"
                     + (f"; it was run on {len(pinn['instances'])} test instances, {pinn['iters']} Adam steps each, "
                        f"{pinn['hardware']}." if pinn else "."))
