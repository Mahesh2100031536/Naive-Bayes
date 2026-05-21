import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model
model = pickle.load(
    open("model.pkl", "rb")
)

vectorizer = pickle.load(
    open("vectorizer.pkl", "rb")
)

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text)

    words = text.split()

    words = [
        w
        for w in words
        if w not in stop_words
    ]

    words = [
        lemmatizer.lemmatize(w)
        for w in words
    ]

    return " ".join(words)


def predict(text):

    cleaned = clean_text(text)

    vector = vectorizer.transform(
        [cleaned]
    )

    pred = model.predict(
        vector
    )[0]

    return (
        "Positive 😊"
        if pred == 1
        else "Negative 😞"
    )


st.title(
    "Restaurant Sentiment Analysis"
)

review = st.text_area(
    "Enter Review"
)

if st.button("Predict"):

    result = predict(review)

    st.success(result)              