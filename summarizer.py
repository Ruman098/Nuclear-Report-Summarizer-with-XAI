import streamlit as st
import numpy as np
import torch
from transformers import pipeline
import shap

# Load a text summarization pipeline
# model_name can be overridden for smaller/larger models
@st.cache_resource
def load_summarizer(model_name="t5-small"):
    return pipeline("summarization", model=model_name)

# Summarize a single document
# Returns the summary string

def summarize(text, summarizer, max_length=150, min_length=40):
    result = summarizer(text, max_length=max_length, min_length=min_length)
    return result[0]["summary_text"]

# Explain summary predictions with SHAP
# Returns a SHAP Values object for plotting

# Explain summary predictions with real SHAP values and readable tokens
def explain(text, summarizer):
    tokenizer = summarizer.tokenizer
    model = summarizer.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Define a wrapper function to return scalar similarity between summary and model output
    def predict_fn(inputs):
        scores = []
        for i in range(len(inputs)):
            input_text = inputs[i]
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
            output_ids = model.generate(input_ids, max_length=100)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Score: ROUGE-like similarity with original summary (simplified)
            baseline_summary = summarize(text, summarizer)
            score = simple_similarity(output_text, baseline_summary)
            scores.append(score)
        return np.array(scores)

    # Merge tokens into readable words
    tokens = tokenizer.tokenize(text)
    readable_tokens = merge_tokens(tokens)

    # SHAP explainer
    explainer = shap.Explainer(predict_fn, shap.maskers.Text(tokenizer))
    shap_values = explainer([text], max_evals=30)

    return shap_values[0]  # First example


# Merge subword tokens back to words
def merge_tokens(tokens):
    words = []
    current_word = ""
    for tok in tokens:
        tok = tok.replace("▁", " ")  # T5/SentencePiece marker
        if tok.startswith(" "):  # new word
            if current_word:
                words.append(current_word)
            current_word = tok.strip()
        else:
            current_word += tok
    if current_word:
        words.append(current_word)
    return words


# Quick similarity metric for summaries (simple version)
def simple_similarity(pred, ref):
    pred_words = set(pred.lower().split())
    ref_words = set(ref.lower().split())
    return len(pred_words & ref_words) / max(1, len(ref_words))