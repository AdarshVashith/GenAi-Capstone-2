"""Utility functions for the Streamlit UI."""

import joblib
import streamlit as st
from fpdf import FPDF
from farm_advisor.config import LABEL_ENCODERS_PATH


@st.cache_resource
def load_label_encoders():
    """Load label encoders for populating dropdown options."""
    return joblib.load(LABEL_ENCODERS_PATH)


def create_pdf(text: str) -> bytes:
    """Generate a simple PDF from the advisory text."""
    pdf = FPDF()
    pdf.add_page()
    # Use a standard font. Note: fpdf doesn't support full markdown,
    # so we strip # and * for the PDF version to keep it clean.
    pdf.set_font("Arial", size=12)

    # Handle utf-8 characters by replacing or ignoring common ones if needed
    clean_text = text.replace("’", "'").replace("–", "-")

    pdf.multi_cell(0, 10, clean_text)
    return bytes(pdf.output())


def render_risk_badge(risk_level: str) -> None:
    """Render the risk band with a Streamlit status callout."""
    if risk_level == "Low":
        st.success("Risk Level: Low")
    elif risk_level == "Medium":
        st.warning("Risk Level: Medium")
    else:
        st.error("Risk Level: High")
