# app.py
import os
import shutil
import json
import pickle
import numpy as np
import streamlit as st

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------------------
# 1. NLTK setup (run once; comment out if already downloaded)
# -------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------------------------------------------------
# 2. Text cleaning – copy from your notebook (final version)
# -------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    # keep only letters (if your final version used digits too, adjust this)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# -------------------------------------------------------------------
# 3. Load tokenizer, meta, and model
# -------------------------------------------------------------------
with open("tokenizer_basic.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("meta_basic.json", "r") as f:
    meta = json.load(f)

categories = meta["categories"]
MAX_LEN = meta["max_len"]

# ⚠️ CHANGE THIS to your actual model filename
MODEL_PATH = "news_basic_dl_model.h5"  # or .h5 or whatever you used
model = load_model(MODEL_PATH)

# -------------------------------------------------------------------
# 4. Prediction helper
# -------------------------------------------------------------------
def predict_document(text: str):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = categories[pred_idx]
    return pred_label, probs

# -------------------------------------------------------------------
# 5. Optional: folder-based sorting (bulk classification)
# -------------------------------------------------------------------
def sort_text_files_in_folder(input_folder: str, output_root: str):
    results = []
    os.makedirs(output_root, exist_ok=True)

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(".txt"):
            continue

        src_path = os.path.join(input_folder, fname)
        with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        label, _ = predict_document(text)

        target_dir = os.path.join(output_root, label)
        os.makedirs(target_dir, exist_ok=True)

        dst_path = os.path.join(target_dir, fname)
        shutil.move(src_path, dst_path)

        results.append((fname, label, dst_path))

    return results

# -------------------------------------------------------------------
# 6. Streamlit UI
# -------------------------------------------------------------------
st.title("Document Classifier (NLP)")
st.write("Deep learning–based classifier for plain text documents.")

mode = st.sidebar.selectbox(
    "Mode",
    ["Single text / file", "Sort a folder of .txt files"]
)

# ------------------------- Mode 1: Single text / file ------------------------
if mode == "Single text / file":
    submode = st.radio("Input type", ["Type / paste text", "Upload .txt file"])

    if submode == "Type / paste text":
        user_text = st.text_area("Enter your document text here", height=200)

        if st.button("Classify"):
            if not user_text.strip():
                st.warning("Please enter some text.")
            else:
                label, probs = predict_document(user_text)
                st.success(f"Predicted category: **{label}**")

                st.subheader("Class probabilities")
                for cat, p in zip(categories, probs):
                    st.write(f"- {cat}: {p:.3f}")

    else:
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
        if uploaded_file is not None and st.button("Classify uploaded file"):
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            label, probs = predict_document(text)
            st.success(f"Predicted category: **{label}**")

            st.subheader("Class probabilities")
            for cat, p in zip(categories, probs):
                st.write(f"- {cat}: {p:.3f}")

# --------------------- Mode 2: Sort a folder of .txt files -------------------
else:
    st.write("This will classify all `.txt` files in a folder and move them into")
    st.write("subfolders named after the predicted category.")

    input_folder = st.text_input("Input folder path (where your .txt files are)")
    output_root = st.text_input(
        "Output root folder (sorted files will be placed here)",
        value="sorted_documents"
    )

    if st.button("Sort folder"):
        if not os.path.isdir(input_folder):
            st.error("Input folder does not exist.")
        else:
            results = sort_text_files_in_folder(input_folder, output_root)
            if not results:
                st.warning("No .txt files found in the input folder.")
            else:
                st.success(f"Sorted {len(results)} files.")
                st.write("Sample results:")
                for fname, label, dst in results[:20]:
                    st.write(f"- `{fname}` → **{label}** → `{dst}`")
