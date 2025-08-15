import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load model and vectorizer
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'best_model.joblib')
model = joblib.load(model_path)


# Streamlit page
st.title("Email Spam Classification")

text_input = st.text_input('Enter your email')

def text_transformer(text):
    text = text.lower()
    text = nltk.word_tokenize(text) #seprate the words
    y = []
    for i in text:
        if i.isalnum():  #alnum --> alpha numeric only
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

if st.button('Predict'):
    transformed_text = text_transformer(text_input)  
    final_transformed_text = cv.transform([transformed_text])
    prediction = model.predict(final_transformed_text)[0]
    
    if prediction == 0:
        st.success('Not Spam ✅')
    else:
        st.error('Spam 🚫')

