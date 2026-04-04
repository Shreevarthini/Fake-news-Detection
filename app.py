import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
st.set_page_config(page_title="AI Fake News Detector", page_icon="📰")

@st.cache_resource
def load_model():
    model_path = "./models/fake_news_model"
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

st.title("AI Fake News Detector")
st.write("Paste a news headline or article snippet below to check its authenticity.")

user_input = st.text_area("News Content:", height=200, placeholder="Enter text here...")

if st.button("Check Authenticity"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing linguistic patterns..."):
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities).item()
                confidence = probabilities[0][prediction].item()
            st.divider()
            if prediction == 1: 
                st.error(f"POTENTIAL FAKE NEWS DETECTED")
                st.metric("Probability of Fake", f"{confidence:.2%}")
                st.write(" This content matches linguistic patterns common in misinformation.")
            else:
                st.success(f"LIKELY GENUINE NEWS")
                st.metric("Authenticity Confidence", f"{confidence:.2%}")
                st.write("This content appears to be consistent with standard journalistic reporting.")
st.sidebar.header("About the Model")
st.sidebar.info("""
This app uses a **DistilBERT** Transformer model fine-tuned on the WELFake dataset. 
It analyzes the 'contextual relationship' between words rather than just looking for keywords.
""")