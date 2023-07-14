import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text (text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():   #i> isalnum sees if the character is alphanumeric
            y.append(i)
    text = y[:]  
    y.clear()    #->we did this beacus string are immutable . so the get exact copy use this slicing method
    for i in text :
        if i not in stopwords.words('english') and i not in string.punctuation :
            y.append(i)    
    text = y[:]
    y.clear()
    for i in text :
        y.append(ps.stem(i))            
    return " ".join(y)   #-> to return as a string  
     
tfidf  = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
# preprocessing

# transforming message
input_msg = st.text_area("Enter your message")
if st.button('Predict'):
   transformed_sms = transform_text(input_msg)
   #vectorizing
   vector_input = tfidf.transform([transformed_sms])
   #predicitng
   result = model.predict(vector_input)[0]
   #display
   if result == 1 :
       st.header("Spam")
   else :
       st.header("Not Spam")
      



