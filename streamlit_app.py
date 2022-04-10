from turtle import color
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

st.title("Spam Filter - Course Project")
st.header("GCIS 523 - Statistical Computing")
st.text("Team 1: Purevdorj Enkhjargal, Dinesh Siripurapu")

input_sms = st.text_area("Enter the text")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    return " ".join([ps.stem(i) for i in text])

tfidf = pickle.load(open('Vectorizer.pkl','rb'))
model = pickle.load(open('SpamDetectorModelNB.pkl','rb'))

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    
    if result:
        st.header("Spam")
    else:
        st.header("Not Spam")