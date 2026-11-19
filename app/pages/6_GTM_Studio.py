import pandas as pd
import streamlit as st
from common import honesty_note, results, setup  # noqa: I001

from physicsveto.gtm import llm
from physicsveto.gtm.content import COMPETITIVE, MOCK, OBJECTIONS, VERTICALS, solution_brief, talk_track

setup("Go-to-Market Studio", "🎯")
st.title("🎯 Go-to-Market Studio")
st.markdown("The assets a partner or sales team needs to take PhysicsVeto to a specific industry.")
if MOCK:
    st.info("Mock mode: all text on this page is pre-written template text. Measured numbers are inserted from "
            "results/.")

s = results("benchmark_summary.json")
vertical = st.selectbox("Industry vertical", list(VERTICALS))
v = VERTICALS[vertical]

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Personas and pains")
    for name, pain in v["personas"]:
        st.markdown(f"**{name}**  \n{pain}")
with c2:
    st.markdown("#### Value propositions")
    for x in v["value"]:
        st.markdown(f"- {x}")
    st.caption(v["case"])

st.markdown("#### Objection handling")
for q, a in OBJECTIONS:
    with st.expander(f"“{q}”"):
        st.markdown(a)

st.markdown("#### Competitive landscape")
st.dataframe(pd.DataFrame(COMPETITIVE, columns=["Category", "Who", "Strengths", "Typical gap"]),
             hide_index=True, width="stretch")
honesty_note("Written at category level and kept neutral. Verify any vendor-specific claim before external use.")

brief = solution_brief(vertical, s)
track = talk_track(vertical, s)
st.markdown("#### Downloads")
d1, d2 = st.columns(2)
d1.download_button("📄 One-page solution brief (.md)", brief, file_name=f"PhysicsVeto_brief_{vertical.split()[0]}.md",
                   mime="text/markdown", width="stretch")
d2.download_button("🎤 5-minute demo talk track (.md)", track, file_name=f"PhysicsVeto_talk_track_{vertical.split()[0]}.md",
                   mime="text/markdown", width="stretch")
t1, t2 = st.tabs(["Solution brief preview", "Talk track preview"])
t1.markdown(brief)
t2.markdown(track)

st.markdown("#### Tailor with an LLM (optional)")
ok, why = llm.available()
if not ok:
    st.caption(f"LLM mode is off: {why} Set PHYSICSVETO_MOCK=0 and ANTHROPIC_API_KEY to enable it. "
               "The templates above work without it.")
else:
    st.caption(f"Drafts with Claude ({llm.MODEL}, server-side fallbacks on). The model only sees the template text "
               "and the measured numbers from results/, and is told not to add numbers of its own. Review before use.")
    l1, l2 = st.columns(2)
    task = l1.selectbox("What to draft", list(llm.TASKS), format_func=lambda t: t.replace("_", " ").capitalize())
    audience = l2.text_input("Audience", "VP of simulation at a mid-size engineering firm")
    if st.button("Draft with Claude", type="primary"):
        measured = {k: v for k, v in (s or {}).items() if k != "per_mode"}
        src = track if task == "talk_track" else brief
        with st.spinner("Drafting..."):
            try:
                r = llm.generate(task, vertical, audience, src, measured)
                st.session_state["llm_draft"] = r
            except RuntimeError as e:
                st.warning(str(e))
    r = st.session_state.get("llm_draft")
    if r:
        st.info(f"LLM-drafted by {r.model}. Check every number against results/ before sharing. {r.note}")
        st.markdown(r.text)
        st.download_button("Download draft (.md)", r.text, file_name="PhysicsVeto_llm_draft.md", mime="text/markdown")
