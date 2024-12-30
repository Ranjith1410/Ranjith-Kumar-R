import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open('spam_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit App
st.title("Spam Email Classifier")
st.write("Enter a message below to check if it is Spam or Ham.")

# Input text
user_input = st.text_area("Message")

if st.button("Classify"):
    # Preprocess and classify input
    input_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(input_tfidf)[0] 
    label = "Spam" if prediction == 1 else "Ham"
    st.write(f"The message is classified as: **{label}**")