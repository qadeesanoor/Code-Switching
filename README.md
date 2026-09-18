# Code Switching NLP 

## About the Project

People in Pakistan commonly mix Roman Urdu and English in everyday digital communication. A sentence such as:

> **“Aaj mera mood nahi hai for anything.”**

contains words from multiple language sources, yet most NLP systems are designed and trained primarily on clean, single-language text.

This project creates a labeled Roman Urdu–English code-switching dataset and fine-tunes **XLM-RoBERTa** to identify the language category of each word. Every word is labeled as **URD, ENG, or MIX**, providing a foundation for NLP applications that need to understand how Pakistanis actually communicate online.

---

## Why This Matters

Roman Urdu is widely used in WhatsApp messages, social media posts, comments, chats, and other informal communication, but it is much less supported by NLP resources than standard English or Urdu.

Code-switching makes language processing harder because a single sentence can contain Urdu and English words, while some English-origin words have become so common in Roman Urdu that they are naturally used as part of everyday speech.

This project addresses that problem by creating a word-level labeled dataset that can help future systems understand **mixed Roman Urdu–English text** more accurately.

Potential applications include:

* Sentiment analysis
* Chatbots and conversational AI
* Social media text analysis
* Word-level language identification
* Roman Urdu NLP
* OCR post-processing
* Text normalization and preprocessing

---

## Label Scheme

Each word is assigned one of three labels:

| Label   | Meaning                                                         | Example                        |
| ------- | --------------------------------------------------------------- | ------------------------------ |
| **URD** | Roman Urdu word                                                 | `aaj`, `mera`, `nahi`          |
| **ENG** | English word                                                    | `anything`, `happy`, `project` |
| **MIX** | English-origin word commonly nativized into everyday Roman Urdu | `phone`, `office`, `meeting`   |

---

## Dataset

The dataset contains **151 sentences** with **1,489 word-level entries**.

| Label     |     Count |
| --------- | --------: |
| URD       |       901 |
| ENG       |       488 |
| MIX       |       100 |
| **Total** | **1,489** |

The data is split at the **sentence level** into training and testing sets using an **80/20 split** with `random_state=42`.

---

## How It Works

The project takes a mixed Roman Urdu–English sentence and breaks it into individual words. Each word is given one of three labels: URD, ENG, or MIX. An XLM-RoBERTa model is trained on these examples so it can learn patterns in the different types of words. After training, the model predicts the appropriate label for each word in a new sentence.

---

## Model & Approach

* **Base Model:** `xlm-roberta-base`
* **Task:** Token classification
* **Labels:** URD / ENG / MIX
* **Train-Test Split:** 80/20
* **Random State:** 42
* **Framework:** HuggingFace Transformers
* **Training:** HuggingFace `Trainer` API
* **Evaluation:** scikit-learn per-token classification metrics

The task is **word-level token classification**, not named entity recognition. Therefore, standard per-token precision, recall, and F1 scores are used instead of span-based BIO/NER metrics such as `seqeval`.

---

## Results

The fine-tuned model achieved strong overall performance:

| Metric                  | F1 Score |
| ----------------------- | -------: |
| **Overall Weighted F1** |  **98%** |
| **URD F1**              |  **99%** |
| **ENG F1**              |  **98%** |
| **MIX F1**              |  **89%** |

### Important Note

The **MIX** category contains only **100 examples** in the complete dataset. Therefore, only around 15–20 MIX words appear in the test split, making the **89% MIX F1 score provisional** rather than a stable benchmark.

The overall 98% F1 score should also be interpreted in the context of the relatively small dataset.

---

## Limitations

### Small Dataset

The dataset contains only 151 sentences, which is small compared with the size of a 270M-parameter transformer such as XLM-RoBERTa.

### Class Imbalance

The three labels are not evenly distributed:

* URD: 901
* ENG: 488
* MIX: 100

Because MIX has significantly fewer examples, its performance estimate is less reliable.

### Limited Test Data

A single 80/20 split means the reported scores can be sensitive to which sentences happen to be included in the test set. A larger dataset and cross-validation or multiple evaluation splits would provide a more reliable benchmark.

---

## Project Structure

```text
code-switching-codesaviours-si26-Qadeesa/
│
├── dataset.csv
│
├── models/
│   └── ...
│
├── README.md
├── requirements.txt
└── ...
```

> The exact files may vary depending on the current version of the repository.

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/qadeesanoor/code-switching-codesaviours-si26-Qadeesa.git
cd code-switching-codesaviours-si26-Qadeesa
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

If the repository contains `requirements.txt`:

```bash
pip install -r requirements.txt
```

For a manual installation, install the main dependencies with:

```bash
pip install torch transformers datasets accelerate scikit-learn pandas numpy
```

### 5. Run the Notebook

If the project uses Jupyter Notebook:

```bash
pip install notebook
jupyter notebook
```

Then open the project notebook and run the cells sequentially.

---

## Technologies Used

* **Python**
* **PyTorch**
* **HuggingFace Transformers**
* **XLM-RoBERTa**
* **HuggingFace Datasets**
* **scikit-learn**
* **Pandas**
* **NumPy**
* **Jupyter Notebook**

---

## Future Improvements

* Expand the dataset from 151 sentences to several thousand examples
* Add more diverse Roman Urdu–English text
* Increase the number of MIX examples
* Include spelling variations common in Roman Urdu
* Evaluate the model on a completely independent dataset
* Compare XLM-RoBERTa with other multilingual and language-identification models
* Build a simple web interface for real-time word-level classification

---

## Use directly Model

*Huggingface*
(https://huggingface.co/qadeesanoor/code-switching-codesaviours-si26-qadeesa/tree/main)

---

## Project Goal

The long-term goal is to contribute a useful starting resource for **Roman Urdu–English NLP**, helping future models understand the mixed-language text that people actually use in everyday digital communication.

