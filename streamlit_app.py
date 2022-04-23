import streamlit as st
import pickle
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

st.title("Spam Filter - Course Project")
st.header("GCIS 523 - Statistical Computing")
st.text("Team 1: Purevdorj Enkhjargal, Dinesh Siripurapu")

input_models = st.selectbox(
     'Select model',
     ('Naive Bayes Classifier', 'Random Forest Classifier', 'K-Nearest Nneighbors Classifier', 'Support Vector Machine Classifier'))
input_sms = st.text_area("Enter the text")

ps = PorterStemmer()

def Clean(Text):
    #using regular expression to remove not necessary characters and make lowercase
    clean_text = re.sub('[^a-zA-Z]', ' ', Text).lower()
    return ' '.join(clean_text.split()) #remove multiple whitespaces into single and return

def Tokenize(text):
    return nltk.word_tokenize(text)

def transform_text(text):
    text = Clean(text)
    text = Tokenize(text)

    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    
    return " ".join([ps.stem(i) for i in text])

tfidf = pickle.load(open('Vectorizer.pkl','rb'))
modelMNB = pickle.load(open('SpamDetectorModelMNB.pkl','rb'))
modelRFC = pickle.load(open('SpamDetectorModelRFC.pkl','rb'))
modelKNC = pickle.load(open('SpamDetectorModelKNC.pkl','rb'))
modelSVC = pickle.load(open('SpamDetectorModelSVC.pkl','rb'))

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = ""
    
    if input_models=='Naive Bayes Classifier':
        result = modelMNB.predict(vector_input)[0]
    if input_models=='Random Forest Classifier':
        result = modelRFC.predict(vector_input)[0]
    if input_models=='K-Nearest Nneighbors Classifier':
        result = modelMNB.predict(vector_input)[0]
    if input_models=='Support Vector Machine Classifier':
        result = modelRFC.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")