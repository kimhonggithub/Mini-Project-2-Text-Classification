
from scipy.sparse import csr_matrix, hstack
from gensim.models import Word2Vec
import joblib
import streamlit as st
import pandas as pd
import sys
sys.path.append('./')
from get_more_features import *
from embedding_word import *
from text_preprocess import *

# Load the saved MLP model
mlp_model = joblib.load('Deploy_Streamlit/best_mlp.pkl')  

# Load the Word2Vec model
model = Word2Vec.load("Deploy_Streamlit/word2vec_model.bin")

scaler = joblib.load('Deploy_Streamlit/maxabs_scaler.pkl')  

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

# Multiline text input for user to enter multiple reviews
user_input = st.text_area("Enter multiple reviews (one per line):", height=200)

if st.button("Predict Sentiment"):
    if user_input:
        # Split the user input into individual reviews
        reviews_list = user_input.split('\n')
        sentiment_predictions = predict_sentiment(reviews_list)
        
        # Map numeric predictions to user-friendly labels
        sentiment_labels = ['Positive' if pred == 1 else 'Negative' for pred in sentiment_predictions]
        
        # Display sentiment predictions for each review
        for review, label in zip(reviews_list, sentiment_labels):
            st.write(f"Review: {review} ")
            st.success(f"Sentiment Prediction: {label}")
    else:
        st.warning("Please enter at least one review.")