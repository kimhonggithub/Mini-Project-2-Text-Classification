from nltk.tokenize import word_tokenize
import nltk
import re
import string
import pandas as pd
import contractions
# Define function for text preprocessing for testing
def clean_text(text, is_lower_case=True):
    if pd.isna(text):  # Check for NaN values
        return ""  # Return an empty string for missing values
    
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Replace repetitions of punctuation with a single punctuation
    text = re.sub(r'([' + string.punctuation + r'])\1+', r'\1', text)
    
    # Remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove emoticons
    emoticons = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')
    text = emoticons.sub('', text)
    
    # Expand contractions
    text = contractions.fix(text)
    
    # Remove stop words
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.remove('no')
    stop_words.remove('not')

    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]

    # Join the words back into a cleaned sentence
    cleaned_text = ' '.join(words)
    
    return cleaned_text