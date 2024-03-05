
from scipy.sparse import csr_matrix, hstack
from gensim.models import Word2Vec
from embedding_word import *
from text_preprocess import clean_text  # Assuming you have a function for text cleaning
import joblib
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

# Load the saved MLP model
mlp_model = joblib.load('mlp_model.pkl')  

# Load the Word2Vec model
model = Word2Vec.load("word2vec_model.bin")

scaler = joblib.load('maxabs_scaler.pkl')  

# ... (previous code remains the same)

def predict_sentiment(reviews):
    cleaned_reviews = [clean_text(review) for review in reviews]

    # Convert the reviews to Word2Vec embeddings
    reviews_transformed = text_to_word_embeddings(cleaned_reviews, model)
   
    # Convert numeric features for the reviews
    reviews_numeric_features = pd.DataFrame({
        'count_positive_words': [count_positive_words(review) for review in cleaned_reviews],
        'count_negative_words': [count_negative_words(review) for review in cleaned_reviews],
        'contain_no': [contain_no(review) for review in cleaned_reviews],
        'contain_not': [contain_not(review) for review in cleaned_reviews],
        'contain_exclamation': [contain_exclamation(review) for review in cleaned_reviews],
        'log_review_length': [log_review_length(review) for review in cleaned_reviews],
        'emotion_label': [get_emotion_label(review) for review in cleaned_reviews],
        'sentiment_score': [calculate_sentiment_score(review) for review in cleaned_reviews],
    })

    # Concatenate the word embeddings and numeric features
    input_data = hstack([reviews_transformed, csr_matrix(reviews_numeric_features)])
    
    # Scale the input data using MaxAbsScaler
    scaled_input = scaler.transform(input_data)
    
    # Make predictions using MLP model
    predictions = mlp_model.predict(scaled_input)
    
    return predictions

# Streamlit UI
st.title("Sentiment Analysis App")

# Input text box for user to enter reviews
user_input = st.text_area("Enter reviews (one per line):")

if st.button("Predict Sentiment"):
    if user_input:
        reviews_list = user_input.split('\n')
        sentiment_predictions = predict_sentiment(reviews_list)
        
        # Map numeric predictions to user-friendly labels
        sentiment_labels = ['Positive' if pred == 1 else 'Negative' for pred in sentiment_predictions]
        
        for review, label in zip(reviews_list, sentiment_labels):
            
            st.write(f"Review: {review} ")
            st.success(f"Sentiment Predictions: {label}")
    else:
        st.warning("Please enter at least one review.")

