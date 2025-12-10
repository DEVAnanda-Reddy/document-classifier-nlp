# 📄 Document Classifier (NLP | Deep Learning)

A deep learning–based **document classification system** built using **TensorFlow, NLP preprocessing, and Streamlit**.  
This project classifies raw text and `.txt` files into predefined categories and also supports **bulk file sorting into folders** based on predicted labels.

---

## 🚀 Features

- ✅ Classify **typed or pasted text**
- ✅ Upload and classify **.txt files**
- ✅ **Batch document classification** from a folder
- ✅ Automatic **file sorting into category folders**
- ✅ Deep learning–based prediction using a trained neural network
- ✅ Interactive **Streamlit web interface**
- ✅ Displays **class probabilities**

---

## 🧠 Tech Stack

- **Python 3.10**
- **TensorFlow / Keras**
- **NLTK**
- **Streamlit**
- **NumPy & Pandas**
- **Regex for text cleaning**

---

## 📂 Project Structure

```

document-classifier-nlp/
│
├── app.py                          # Streamlit app
├── meta_basic.json                 # Model metadata (categories, max_len)
├── requirements.txt                # Required dependencies
├── Document_Classification.ipynb   # Training notebook
├── basic_dl_doc_classification.ipynb
├── project.ipynb
├── Data/                           # Training data
├── New_Files/                      # New test files
└── Doc calssifier.png              # App preview image

````

---

## ⚙️ Installation & Setup

### 1️⃣ Create Environment (Recommended)
```bash
conda create -n docclass python=3.10 -y
conda activate docclass
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download NLTK Resources (Auto on first run)

The app automatically downloads:

* stopwords
* punkt
* wordnet

---

## ▶️ Run the Streamlit App

```bash
python -m streamlit run app.py
```

Then open the browser link shown in terminal, for example:

```
http://localhost:8501
```

---

## 🗃️ Model & Tokenizer

This project uses:

* A trained deep learning model (`.h5`)
* A saved tokenizer (`.pkl`)

> ⚠️ These files are **not included in the repository** due to size and security.
> You must place your trained model and tokenizer in the project root to run predictions.

---

## 📊 Functional Modes

### ✅ Single Text Classification

* Type or paste text
* Upload `.txt` files
* Get predicted category + probability scores

### ✅ Bulk Folder Classification

* Input a folder path
* Automatically sorts `.txt` files into category folders

---

## 🖼️ Streamlit App Interface Preview

![Streamlit UI Screenshot](https://raw.githubusercontent.com/DEVAnanda-Reddy/document-classifier-nlp/main/UI_ScreenShot.png)


---


---
