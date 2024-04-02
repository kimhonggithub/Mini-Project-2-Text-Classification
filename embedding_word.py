import numpy as np
# Convert text data using Word2Vec embeddings
def text_to_word_embeddings(text_data, word2vec_model):
    # Initialize an empty list to store the word embeddings for each review
    embeddings = []
    
    for review in text_data:
        # Tokenize the review into individual words
        words = review.split()
        # Filter out words not in the Word2Vec vocabulary
        words = [word for word in words if word in word2vec_model]
        # Get the embeddings for each word and take the mean to get the review's embedding
        if words:
            embeddings.append(np.mean(word2vec_model[words], axis=0))
        else:
            # If no words in the vocabulary, return a vector of zeros
            embeddings.append(np.zeros(word2vec_model.vector_size))
            
    return np.array(embeddings)