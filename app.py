    ### Simple app.py file to analyse the user sentiment

## Import Libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load pre-trained model
model = load_model('imdb_simplernn_model.h5')

# Load imdb word-index mapping
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Prepare the user input text for prediction
def preprocess_text(text):
    # Tokenize and convert to sequence of integers
    tokens = text.lower().split()
    encoded_words = [min(word_index.get(word, 2)+3,9999) for word in tokens] # 2 is for unknown words, offset by 3
    padded_review = pad_sequences([encoded_words], maxlen=500)
    return padded_review

# Make Prediction on input text
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)

    sentiment = "Positive" if prediction[0][0]>0.5 else "Negative"
    return sentiment, prediction[0][0]

## Streamlit app code
import streamlit as st

# Page configuration
st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬", layout="centered")

# Header with emoji
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("### Analyze the sentiment of your movie reviews instantly!")
st.divider()

# Instructions in an info box
st.info("💡 **How it works:** Enter your movie review below and click the button to discover if it's positive or negative!")

# Text input with placeholder
user_input = st.text_area(
    "📝 Enter your movie review here:",
    height=150, max_chars=1000,
    placeholder="Example: This movie was absolutely amazing! The acting was superb and the plot kept me engaged throughout..."
)

st.caption(f"Characters: {len(user_input)}/1000")

# Center the button using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 Predict Sentiment", use_container_width=True)

st.divider()

# Prediction section
if predict_button:
    if user_input.strip():
        with st.spinner("🤔 Analyzing your review..."):
            sentiment, confidence = predict_sentiment(user_input)
        
        # Display results with color coding
        if sentiment == "Positive":
            st.success(f"### ✅ Sentiment: {sentiment}")
            st.balloons()
        else:
            st.error(f"### ❌ Sentiment: {sentiment}")
        
        # Confidence score with progress bar
        st.metric(label="Confidence Score", value=f"{confidence:.2%}")
        st.progress(float(confidence))
        
        # Additional info
        st.caption(f"Raw confidence value: {confidence:.4f}")
    else:
        st.warning("⚠️ Please enter a review before clicking 'Predict Sentiment'")
else:
    st.markdown("👆 **Ready to get started?** Enter your review above and click the button!")

# Footer
st.divider()
st.caption("⚙️ Powered by Simple RNN Model | Made with ❤️ using Streamlit and Deep Learning")
st.caption("© 2025 Piyush Yadav")