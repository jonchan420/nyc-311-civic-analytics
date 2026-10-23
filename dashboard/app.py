"""Streamlit entry point for NYC 311 Civic Analytics: the dashboard and the
Ask 311 assistant, as two pages.

Run from the project root:
    streamlit run dashboard/app.py
"""

import streamlit as st

st.set_page_config(page_title="NYC 311 Civic Analytics", layout="wide")
page = st.navigation([
    st.Page("overview.py", title="Dashboard", icon=":material/insights:", default=True),
    st.Page("ask.py", title="Ask 311", icon=":material/forum:", url_path="ask"),
], position="top")
page.run()
