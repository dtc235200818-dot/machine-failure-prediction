"""CSS dùng chung cho toàn bộ ứng dụng."""
import streamlit as st

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1150px;}
.risk-card {padding: 18px 22px; border-radius: 14px; color: #fff; margin: 6px 0 10px 0;}
.risk-model {font-size: .85rem; opacity: .92; letter-spacing: .02em;}
.risk-title {font-size: 1.9rem; font-weight: 700; line-height: 1.25;}
.risk-sub {font-size: 1.05rem; margin-top: 2px;}
.small-note {font-size: .85rem; opacity: .75;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)
