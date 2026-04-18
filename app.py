"""Streamlit UI for the Farm Advisory Assistant."""

from __future__ import annotations
import joblib
import pandas as pd
import streamlit as st
from fpdf import FPDF

from agent.farm_agent import run_farm_agent, answer_follow_up
from config import LABEL_ENCODERS_PATH


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


def main() -> None:
    """Render the Farm Advisory Assistant UI."""
    st.set_page_config(page_title="Farm Advisory Assistant", layout="centered")
    st.title("Farm Advisory Assistant")
    st.write("Generate a crop-specific advisory using your prediction model and agronomy references.")

    # Initialize session state for persistence
    if "final_state" not in st.session_state:
        st.session_state.final_state = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    try:
        label_encoders = load_label_encoders()
    except FileNotFoundError:
        st.error(
            "Missing model artifacts. Add `models/label_encoders.pkl`, `models/rf_model.pkl`, "
            "and `models/scaler.pkl` before running the advisory UI."
        )
        st.stop()

    area_options = sorted(label_encoders["Area"].classes_.tolist())
    crop_options = sorted(label_encoders["Item"].classes_.tolist())

    with st.sidebar:
        st.header("Farm Inputs")
        area = st.selectbox("Area", area_options)
        item = st.selectbox("Crop/Item", crop_options)
        avg_temp = st.number_input("Avg Temperature", value=25.0)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=800.0)
        pesticide_usage = st.number_input("Pesticide Usage (tonnes)", min_value=0.0, value=50.0)
        submit = st.button("Get Advisory", use_container_width=True)

    if submit:
        farm_data = {
            "Area": area,
            "Item": item,
            "avg_temp": avg_temp,
            "average_rain_fall_mm_per_year": rainfall,
            "pesticides_tonnes": pesticide_usage,
        }
        try:
            with st.spinner("Generating advisory report..."):
                st.session_state.final_state = run_farm_agent(farm_data)
                st.session_state.chat_history = [] # Reset chat for new report
        except Exception as exc:
            st.error(f"Unable to generate advisory: {exc}")
            st.stop()

    # Display results if they exist in state
    if st.session_state.final_state:
        final_state = st.session_state.final_state
        predicted_yield = final_state["yield_prediction"]["predicted_yield"]
        risk_level = final_state["yield_prediction"]["risk_level"]

        st.metric("Predicted Yield", f"{predicted_yield:.2f}")
        render_risk_badge(risk_level)

        with st.expander("Yield Visualization", expanded=True):
            chart_data = pd.DataFrame(
                {
                    "Category": ["Predicted Yield", "Typical Benchmark"],
                    "Yield Value": [predicted_yield, 4.0],
                }
            )
            st.bar_chart(chart_data, x="Category", y="Yield Value", color="#4CAF50")

        st.markdown(final_state["final_report"])

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Export as Markdown",
                data=final_state["final_report"],
                file_name=f"farm_advisory_{item.lower()}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            try:
                pdf_bytes = create_pdf(final_state["final_report"])
                st.download_button(
                    label="Export as PDF",
                    data=pdf_bytes,
                    file_name=f"farm_advisory_{item.lower()}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Could not generate PDF: {e}")

        st.divider()
        st.subheader("Follow-up Questions")
        st.write("Ask the AI advisor more details about this report.")

        # Display history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your farm advisory..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = answer_follow_up(final_state["final_report"], prompt)
                    st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
