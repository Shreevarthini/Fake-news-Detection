# AI-Powered Fake News Detector

A Deep Learning application that classifies news articles as **Real** or **Fake** using a fine-tuned **DistilBERT** Transformer model.

## Project Overview
This project addresses the challenge of online misinformation by analyzing the linguistic patterns and contextual relationships in news headlines and body text. Unlike keyword-based filters, this model understands the nuances of language.

### The Model: DistilBERT
* **Architecture:** Used `distilbert-base-uncased` (a distilled, faster version of BERT).
* **Fine-Tuning:** Trained on the **WELFake Dataset** (72k+ articles) using Google Colab's T4 GPU.
* **Performance:** Achieved a **98.6% F1-Score** on the validation set.

### Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch, Hugging Face Transformers
* **Web Framework:** Streamlit
* **Deployment:** Hugging Face Spaces

## Key Insights
* **Attention Mechanism:** The model uses self-attention to weigh the importance of different words in a sentence, allowing it to detect "sensationalist" or "unstructured" patterns common in fake news.
* **Recall Optimization:** Focused on minimizing False Negatives (Fake news labeled as Real) to ensure public safety.

## How to Run Locally
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`