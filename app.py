# app.py
import os
import shutil
import json
import pickle
import re

import numpy as np
import pandas as pd
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =============================================================================
# 0. Streamlit page config + basic styling
# =============================================================================
st.set_page_config(
    page_title="Document Classifier (NLP)",
    page_icon="📄",
    layout="wide",
)

# Custom CSS for a cleaner look
st.markdown(
    """
    <style>
        .main {
            background-color: #f5f7fb;
        }
        .stApp {
            background-color: #f5f7fb;
        }
        .app-header {
            padding: 1rem 0 0.5rem 0;
        }
        .app-subtitle {
            color: #6c757d;
            font-size: 0.95rem;
        }
        .result-card {
            padding: 1rem 1.2rem;
            border-radius: 0.8rem;
            background-color: #ffffff;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .metric-card {
            padding: 0.8rem 1rem;
            border-radius: 0.8rem;
            background-color: #ffffff;
            box-shadow: 0 1px 6px rgba(0,0,0,0.04);
        }
        .footer-note {
            color: #999999;
            font-size: 0.8rem;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 1. NLTK setup
# =============================================================================
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# =============================================================================
# 2. Text cleaning – same logic as your notebook
# =============================================================================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# =============================================================================
# 3. Load tokenizer, meta, and model
# =============================================================================
with open("tokenizer_basic.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("meta_basic.json", "r") as f:
    meta = json.load(f)

categories = meta["categories"]
MAX_LEN = meta["max_len"]

MODEL_PATH = "news_basic_dl_model.h5"  # make sure this exists
model = load_model(MODEL_PATH)


# =============================================================================
# 4. Prediction helpers
# =============================================================================
def predict_document(raw_text: str):
    cleaned = clean_text(raw_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = categories[pred_idx]
    return pred_label, probs, cleaned


def sort_text_files_in_folder(input_folder: str, output_root: str):
    results = []
    os.makedirs(output_root, exist_ok=True)

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(".txt"):
            continue

        src_path = os.path.join(input_folder, fname)
        with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        label, probs, _ = predict_document(text)

        target_dir = os.path.join(output_root, label)
        os.makedirs(target_dir, exist_ok=True)

        dst_path = os.path.join(target_dir, fname)
        shutil.move(src_path, dst_path)

        results.append(
            {
                "file": fname,
                "predicted_label": label,
                "max_probability": float(np.max(probs)),
                "destination": dst_path,
            }
        )

    return results


# =============================================================================
# 5. Sidebar – mode + info
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    mode = st.radio(
        "Select mode",
        ["Single document", "Batch upload (.txt)", "Local folder sorter"],
    )

    st.markdown("---")
    st.markdown("### ℹ️ Model info")
    st.write(f"**Model path:** `{MODEL_PATH}`")
    st.write(f"**Num classes:** `{len(categories)}`")
    st.write(f"**Max sequence length:** `{MAX_LEN}`")

    st.markdown("---")
    st.caption(
        "Tip: use *Batch upload* to test the model on multiple documents at once."
    )

# =============================================================================
# 6. Header
# =============================================================================
st.markdown(
    """
    <div class="app-header">
        <h1>📄 Document Classifier (NLP)</h1>
        <p class="app-subtitle">
            Deep learning–based classifier for plain text documents. 
            Paste text, upload files, or sort entire folders.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top metrics row
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown('<div class="metric-card">✅ Model loaded</div>', unsafe_allow_html=True)
with col_m2:
    st.markdown(
        f'<div class="metric-card">📚 Classes: <b>{len(categories)}</b></div>',
        unsafe_allow_html=True,
    )
with col_m3:
    st.markdown(
        f'<div class="metric-card">🧵 Max tokens: <b>{MAX_LEN}</b></div>',
        unsafe_allow_html=True,
    )

st.markdown("")


# =============================================================================
# 7. Single document mode
# =============================================================================
if mode == "Single document":
    tab1, tab2 = st.tabs(["🔤 Type / paste text", "📁 Upload .txt file"])

    # Initialise prediction history in session state
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # ------------- Tab 1: Text area -------------
    with tab1:
        left, right = st.columns([1.3, 1])

        with left:
            st.subheader("Input document")
            user_text = st.text_area(
                "Enter your document text here",
                height=250,
                placeholder="Paste any article, news, review, or document here...",
            )
            show_cleaned = st.checkbox("Show cleaned text preview", value=False)

            if st.button("🔍 Classify text", type="primary"):
                if not user_text.strip():
                    st.warning("Please enter some text.")
                else:
                    label, probs, cleaned = predict_document(user_text)

                    # Save to history
                    st.session_state["history"].append(
                        {"text": user_text[:120] + ("..." if len(user_text) > 120 else ""),
                         "label": label,
                         "confidence": float(np.max(probs))}
                    )

                    with right:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.subheader("Prediction")
                        st.success(f"**Predicted category:** {label}")

                        probs_df = (
                            pd.DataFrame(
                                {"Category": categories, "Probability": probs}
                            )
                            .sort_values("Probability", ascending=False)
                            .reset_index(drop=True)
                        )

                        st.write("Top classes:")
                        st.dataframe(
                            probs_df.head(5),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.write("Class probability distribution")
                        st.bar_chart(
                            probs_df.set_index("Category"),
                            use_container_width=True,
                        )

                        st.markdown("</div>", unsafe_allow_html=True)

                    if show_cleaned:
                        with st.expander("Show cleaned / preprocessed text"):
                            st.text(cleaned)

        with right:
            # If no prediction yet, show placeholder card
            if not st.session_state["history"]:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader("Prediction")
                st.info(
                    "Run a classification to see probabilities, charts, and details here."
                )
                st.markdown("</div>", unsafe_allow_html=True)

    # ------------- Tab 2: Single file upload -------------
    with tab2:
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            if st.button("📄 Classify file", type="primary"):
                label, probs, cleaned = predict_document(text)

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader("Prediction")
                st.success(f"**Predicted category:** {label}")

                probs_df = (
                    pd.DataFrame({"Category": categories, "Probability": probs})
                    .sort_values("Probability", ascending=False)
                    .reset_index(drop=True)
                )

                st.write("Top classes:")
                st.dataframe(
                    probs_df.head(5),
                    use_container_width=True,
                    hide_index=True,
                )

                st.bar_chart(
                    probs_df.set_index("Category"),
                    use_container_width=True,
                )

                with st.expander("Show cleaned / preprocessed text"):
                    st.text(cleaned)

                st.markdown("</div>", unsafe_allow_html=True)

    # ------------- Prediction history -------------
    if st.session_state["history"]:
        st.markdown("### 🕒 Recent predictions")
        hist_df = pd.DataFrame(st.session_state["history"][-10:][::-1])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


# =============================================================================
# 8. Batch upload mode
# =============================================================================
elif mode == "Batch upload (.txt)":
    st.subheader("Batch classification of uploaded .txt files")

    uploaded_files = st.file_uploader(
        "Upload one or more .txt files",
        type=["txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("🚀 Run batch classification", type="primary"):
        rows = []
        for f in uploaded_files:
            text = f.read().decode("utf-8", errors="ignore")
            label, probs, _ = predict_document(text)
            rows.append(
                {
                    "file": f.name,
                    "predicted_label": label,
                    "confidence": float(np.max(probs)),
                }
            )

        df = pd.DataFrame(rows)
        st.markdown("#### Results")
        st.dataframe(df, use_container_width=True, hide_index=True)

        summary = df.groupby("predicted_label")["file"].count().reset_index()
        summary.columns = ["Category", "Count"]

        st.markdown("#### Category distribution")
        st.bar_chart(summary.set_index("Category"), use_container_width=True)


# =============================================================================
# 9. Local folder sorter mode
# =============================================================================
else:
    st.subheader("Local folder sorter")

    st.write(
        "Classify all `.txt` files inside a folder and automatically move them "
        "into subfolders named after the predicted category."
    )

    col1, col2 = st.columns(2)
    with col1:
        input_folder = st.text_input(
            "Input folder path (where your .txt files are)",
            help="Example: C:/Users/devan/AI-ML/raw_documents",
        )
    with col2:
        output_root = st.text_input(
            "Output root folder (sorted files will be placed here)",
            value="sorted_documents",
            help="Will be created if it doesn't exist.",
        )

    if st.button("📂 Sort local folder", type="primary"):
        if not os.path.isdir(input_folder):
            st.error("Input folder does not exist.")
        else:
            results = sort_text_files_in_folder(input_folder, output_root)
            if not results:
                st.warning("No .txt files found in the input folder.")
            else:
                df = pd.DataFrame(results)
                st.success(f"Sorted {len(df)} files.")

                st.markdown("#### Sample of sorted files")
                st.dataframe(df.head(20), use_container_width=True, hide_index=True)

                dist = df.groupby("predicted_label")["file"].count().reset_index()
                dist.columns = ["Category", "Count"]

                st.markdown("#### Category distribution")
                st.bar_chart(dist.set_index("Category"), use_container_width=True)

# =============================================================================
# 10. Footer
# =============================================================================
st.markdown(
    '<p class="footer-note">Built with Streamlit · Deep learning document classifier</p>',
    unsafe_allow_html=True,
)
