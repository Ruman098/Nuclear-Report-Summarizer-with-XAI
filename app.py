import streamlit as st
from summarizer import load_summarizer, summarize, explain
from PyPDF2 import PdfReader
import streamlit.components.v1 as components
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Nuclear Summarizer XAI", layout="wide")

# Load the summarization pipeline and XAI explainer
summarizer = load_summarizer(model_name="t5-small")

st.title("🔬 Nuclear Report Summarizer + XAI")

uploaded = st.file_uploader("Upload nuclear report (PDF or TXT)", type=["pdf", "txt"])
if not uploaded:
    st.info("Please upload a PDF or TXT file to summarize.")
    st.stop()

# Extract text from upload
if uploaded.type == "application/pdf":
    reader = PdfReader(uploaded)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
else:
    text = uploaded.read().decode("utf-8")

# Generate summary
with st.spinner("Generating summary..."):
    summary_text = summarize(text, summarizer)

st.subheader("Summary")
st.write(summary_text)

short_text = text[:300]
# Explainability with SHAP
with st.spinner("Computing explanations..."):
    shap_values = explain(text, summarizer)

st.subheader("Which parts drove the summary?")
html = shap.plots.text(shap_values, display=False)
components.html(html, height=400, scrolling=True)

with st.expander("🔍 How to interpret this visualization?", expanded=False):
    st.markdown("""
    - 🟥 **Red highlights** mean those words **increased** the model's confidence in producing the summary.
    - 🟦 **Blue highlights** mean those words **decreased** the model's confidence.
    - The **intensity** of the color reflects how strong the influence was.
    """)
