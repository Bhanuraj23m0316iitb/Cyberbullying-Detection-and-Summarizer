import re
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import string

# Introduction menu (for console use, optional if using Streamlit)
def intro():
    print("""
    ********************************************** Choice Menu ************************************************
    
        ************** Text Summarizer and Cyberbullying Detection **************
        
        1. Check the summary of the Text Input
        2. Check whether the language was Offensive or not
        
    """)

# Cleaning the text
def clean(text, stopword):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    return " ".join(text)

# Word frequency counter for summarization
def word_freq_counter(doc, word_freq):
    stop_words = list(STOP_WORDS)
    for word in doc:
        if word.text.lower() not in stop_words and word.text not in punctuation:
            word_lower = word.text.lower()
            word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
    return word_freq

# Sentence scoring
def sentence_score(sent_token, sent_score, word_freq):
    for sent in sent_token:
        for word in sent:
            word_lower = word.text.lower()
            if word_lower in word_freq:
                sent_score[sent] = sent_score.get(sent, 0) + word_freq[word_lower]
    return sent_score
