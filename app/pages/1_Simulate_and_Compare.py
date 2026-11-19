import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import (  # noqa: I001
    MODE_LABELS,
    case_cfg,
    heat,
    honesty_note,
    make_input,
    p0,
    run_all,
    setup,
    surrogates,
    velocity_fig,
)

setup("Simulate & Compare", "🧪")
st.title("🧪 Simulate & Compare")
st.markdown("Run the finite-difference reference solver and both AI surrogates on the same input, side by side.")

case, cfg = case_cfg()
if not surrogates():
    st.error("No trained weights in weights/. Run `make train`.")
    st.stop()

with st.sidebar:
    st.header("Input")
    mode = st.selectbox("Geology", list(MODE_LABELS), format_func=MODE_LABELS.get, key="sim_mode")
    seed = st.number_input("Random seed", 0, 10_000, 3, key="sim_seed")
    src_x = None
    if mode == "in_distribution":
        lo, hi = case["source_x_range"]
        src_x = st.slider("Shot position (column)", lo, hi, (lo + hi) // 2)
    st.button("🎲 New random model", on_click=lambda: st.session_state.update(sim_seed=int(np.random.randint(10_000))))

v, src = make_input(mode, int(seed), src_x)
out = run_all(v, src)
ref, fno, unet = out["ref"], out["fno"], out.get("unet")
anchors = case["anchors"]
times_ms = [a * cfg.dt * 1e3 for a in anchors]

c1, c2, c3, c4 = st.columns(4)
c1.metric("FD solver", f"{out['t_fd']:.1f} ms", f"{cfg.nt} time steps", delta_color="off")
c2.metric("FNO surrogate", f"{out['t_fno']:.1f} ms", "single case, 1 thread", delta_color="off")
c3.metric("Speedup (this run)", f"{out['t_fd'] / out['t_fno']:.2f}×", "FNO vs FD, CPU, 64×64", delta_color="off")
err = np.linalg.norm(p0(fno) - p0(ref)) / np.linalg.norm(p0(ref))
c4.metric("FNO error vs solver", f"{100 * err:.1f}%", "relative L2, all snapshots", delta_color="off")
honesty_note("Live wall-clock on the server running this app. At 64×64 the solver is cheap; see the Benchmark Lab "
             "for how the gap changes with resolution.")

k = st.select_slider("Snapshot time", options=list(range(len(anchors))),
                     format_func=lambda i: f"{times_ms[i]:.0f} ms", value=4)
pr, pf = p0(ref)[k], p0(fno)[k]
lim = float(np.abs(pr).max())
cols = st.columns(4)
cols[0].plotly_chart(velocity_fig(v, src), width="stretch")
cols[1].plotly_chart(heat(pr, "FD solver (reference)", zmid=0, zmin=-lim, zmax=lim), width="stretch")
cols[2].plotly_chart(heat(pf, "FNO surrogate", zmid=0, zmin=-lim, zmax=lim), width="stretch")
err_map = np.abs(pf - pr)
cols[3].plotly_chart(heat(err_map, f"|FNO − FD| (max {100 * err_map.max() / lim:.1f}% of peak)", colorscale="Magma",
                          zmin=0, zmax=float(err_map.max()) + 1e-9),
                     width="stretch")

st.markdown("#### Wavefield animation")
which = st.radio("Show", ["FD solver", "FNO", "U-Net"] if unet is not None else ["FD solver", "FNO"],
                 horizontal=True)
src_pred = {"FD solver": ref, "FNO": fno, "U-Net": unet}[which]
frames = p0(src_pred)
lim_all = float(np.abs(p0(ref)).max()) * 0.6
fig = go.Figure(
    data=[go.Heatmap(z=frames[0], colorscale="RdBu", zmid=0, zmin=-lim_all, zmax=lim_all, showscale=False)],
    frames=[go.Frame(data=[go.Heatmap(z=f)], name=f"{t:.0f} ms") for f, t in zip(frames, times_ms, strict=True)],
)
fig.update_layout(
    template="plotly_dark", height=460, margin={"l": 10, "r": 10, "t": 10, "b": 10},
    paper_bgcolor="rgba(0,0,0,0)",
    updatemenus=[{"type": "buttons", "showactive": False, "x": 0.02, "y": 1.08,
                  "buttons": [{"label": "▶ Play", "method": "animate",
                               "args": [None, {"frame": {"duration": 450, "redraw": True}, "fromcurrent": True}]}]}],
    sliders=[{"steps": [{"label": f"{t:.0f} ms", "method": "animate",
                         "args": [[f"{t:.0f} ms"], {"mode": "immediate", "frame": {"redraw": True}}]}
                        for t in times_ms], "x": 0.1, "len": 0.85}],
)
fig.update_yaxes(autorange="reversed", showticklabels=False, scaleanchor="x")
fig.update_xaxes(showticklabels=False)
st.plotly_chart(fig, width="stretch")

if unet is not None:
    st.markdown("#### Error over time")
    e_f = [np.linalg.norm(p0(fno)[i] - p0(ref)[i]) / np.linalg.norm(p0(ref)[i]) for i in range(len(anchors))]
    e_u = [np.linalg.norm(p0(unet)[i] - p0(ref)[i]) / np.linalg.norm(p0(ref)[i]) for i in range(len(anchors))]
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=times_ms, y=e_f, name="FNO", line={"color": "#76b900"}))
    f2.add_trace(go.Scatter(x=times_ms, y=e_u, name="U-Net", line={"color": "#8fa3ff"}))
    f2.update_layout(template="plotly_dark", height=280, yaxis_title="relative L2 error", xaxis_title="time (ms)",
                     yaxis_tickformat=".0%", margin={"l": 10, "r": 10, "t": 10, "b": 10},
                     paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(f2, width="stretch")
    honesty_note("Error usually grows with time as the wavefield becomes more complex. This is one reason to "
                 "check every prediction rather than trust an average accuracy number.")
