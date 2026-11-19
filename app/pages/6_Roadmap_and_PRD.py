from common import ROOT, setup  # noqa: I001

import streamlit as st

setup("Roadmap & PRD", "🗺️")
st.markdown((ROOT / "docs" / "PRD.md").read_text(encoding="utf-8"))
