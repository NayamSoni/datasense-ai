import streamlit as st
from pathlib import Path


def load_theme():

    # Resolve beside this Python file instead of relative to the terminal's
    # current directory. Starting Streamlit with an absolute app path should
    # never make the entire interface disappear because CSS was not found.
    style_path = Path(__file__).resolve().with_name("styles.css")

    if not style_path.exists():
        st.warning(
            f"Theme file not found: {style_path}. "
            "Rename the downloaded stylesheet to styles.css."
        )
        return

    with style_path.open(encoding="utf-8") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
