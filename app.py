# Step 1: Import Libraries and Load the Model
import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from PIL import Image

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model_path = 'simple_rnn_imdb_v2.h5'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure the model file is in the correct directory.")
else:
    model = load_model(model_path)

    # Step 2: Helper Functions
    # Function to decode reviews
    def decode_review(encoded_review):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

    # Function to preprocess user input
    def preprocess_text(text):
        words = text.lower().split()
        encoded_review = [word_index.get(word, 2) + 3 for word in words]
        padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
        return padded_review

    # Streamlit app
    st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎥", layout="centered")

    # Add a header image
    image = Image.open('./Images/header.jpeg')  # Path to the header image
    image = image.resize((500, 200))  # Resize the image to make it smaller
    st.image(image, use_column_width=False)

    st.title('IMDB Movie Review Sentiment Analysis 🎥')
    st.write('Enter a movie review to classify it as positive or negative.')

    # Add a brief description
    st.markdown("""
    This app uses a Recurrent Neural Network (RNN) to analyze the sentiment of movie reviews. The model was trained on the IMDB dataset, which contains 50,000 movie reviews.
    """)

    # Layout the input and output sections
    col1, col2 = st.columns(2)

    with col1:
        user_input = st.text_area('Movie Review', height=200)

    with col2:
        if st.button('Classify'):
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
            prediction_score = float(prediction[0][0])  # Convert to standard Python float

            # Display the result
            st.markdown(f"### Sentiment: **{sentiment}**")
            st.markdown(f"#### Prediction Score: **{prediction_score:.2f}**")
            st.progress(prediction_score)
        else:
            st.write('Please enter a movie review.')

    # Add a footer with credits
    st.markdown("""
    ---
    ### Credits

    - **Model and Code:** Ankit Wahane
    - **Dataset:** [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)
    - **Header Image:** Generated using OpenAI's DALL-E
    This project is developed as part of a sentiment analysis demonstration using a Recurrent Neural Network (RNN) model.
    """)
