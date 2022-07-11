# Contents of ~/my_app/main_page.py
import streamlit as st

st.set_page_config(
    page_title="Persian OCR",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📖 Persian OCR")
st.markdown(
    """## Persian OCR allows users to **scan** documents and extract **text** from scanned image.
### How to use:
 - **First** Scan Your Image Using **Scan** Page
 - **Second** Extract Text From Your Scanned Image Using **OCR** Page
"""
)
