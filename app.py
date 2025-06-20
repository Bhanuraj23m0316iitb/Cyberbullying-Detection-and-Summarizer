import streamlit as st
from definations import *
import pandas as pd
import numpy as np
import re
import spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from heapq import nlargest

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load and clean dataset
df = pd.read_csv("twitter_data.csv")
df.dropna(subset=['tweet', 'labels'], inplace=True)
df['labels'] = df['labels'].astype(int)
df['labels'] = df['labels'].map({
    0: "Hate Speech Detected",
    1: "Offensive language Detected",
    2: "No hate and offensive speech"
})
df = df[['tweet', 'labels']]

# Define stopwords for manual cleaning
stopword = list(spacy.lang.en.stop_words.STOP_WORDS)

# Clean text function
def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    return " ".join(text)

# Clean all tweets
df["tweet"] = df["tweet"].apply(clean)

# Feature extraction using TF-IDF
x = df["tweet"]
y = df["labels"]
cv = TfidfVectorizer(max_features=5000)
x_vector = cv.fit_transform(x)

# Gradient Boosting Classifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
clf.fit(x_vector, y)

# Streamlit UI
st.title("🛡️ Cyberbullying Analyzer with Summary")

choice = st.radio("Choose a function", ["Summarize Text", "Detect Offensiveness"])
user_input = st.text_area("Enter your text here:")

if st.button("Analyze") and user_input.strip() != "":
    if choice == "Summarize Text":
        doc = nlp(user_input)
        word_freq = word_freq_counter(doc, {})
        if word_freq:
            max_freq = max(word_freq.values())
            word_freq = {word: freq / max_freq for word, freq in word_freq.items()}
            sent_token = [sent for sent in doc.sents]
            sent_score = sentence_score(sent_token, {}, word_freq)
            num_lines = max(1, int(len(sent_score) * 0.3))
            summary = nlargest(n=num_lines, iterable=sent_score, key=sent_score.get)

            st.subheader("📄 Summary:")
            for line in summary:
                st.write(line.text)
        else:
            st.warning("Not enough content to summarize.")

    elif choice == "Detect Offensiveness":
        clean_input = clean(user_input)
        vec = cv.transform([clean_input]).toarray()
        probs = clf.predict_proba(vec)[0]
        predicted_label = clf.classes_[np.argmax(probs)]
        confidence = np.max(probs)

        st.subheader("🚨 Prediction:")
        if confidence < 0.5:
            st.info("🟡 The model is unsure – the language appears neutral or ambiguous.")
        elif "No hate" in predicted_label:
            st.success(predicted_label)
        elif "Offensive" in predicted_label:
            st.warning(predicted_label)
        else:
            st.error(predicted_label)

            
     
