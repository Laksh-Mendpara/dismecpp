import json
import spacy
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Load SpaCy's English model
nlp = spacy.load('en_core_web_sm')

# Load the stop words
stop_words = set(stopwords.words('english'))


# Function to clean and tokenize title using SpaCy
def clean_title(title):
    # Remove content inside any kind of brackets and symbols like -, +, #, etc.
    title = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|\<.*?\>', '', title)
    title = re.sub(r'[-+/#]', '', title)

    # Remove extra spaces and tabs
    title = re.sub(r'\s+', ' ', title).strip()

    # Process the title using SpaCy
    doc = nlp(title)

    # Tokenize and convert to lowercase, remove stop words and punctuation
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and token.text.isalpha()]
    return tokens


# Load the pre-saved vocabulary from .pkl file
with open('vocabulary_large.pkl', 'rb') as vocab_file:
    filtered_vocabulary = pickle.load(vocab_file)

# Load the JSON data and preprocess the 'content' field
data_points = []
target_labels = []
with open('tst.json', 'r') as f:
    c = 0
    for line in f:
        if c%1000==0:   print(c)
        c += 1
        data_point = json.loads(line)
        title = data_point.get('title', '')
        cleaned_tokens = clean_title(title)
        # Keep only tokens that are in the filtered vocabulary
        final_tokens = [token for token in cleaned_tokens if token in filtered_vocabulary]
        # Join tokens back into a string for TF-IDF processing
        data_points.append(' '.join(final_tokens))
        target_labels.append(data_point['target_ind'])  # Assuming 'target_ind' contains the labels

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(vocabulary=filtered_vocabulary)

# Fit and transform the data to create the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(data_points)

# Save the final data in the required format
num_data_points = tfidf_matrix.shape[0]
num_features = tfidf_matrix.shape[1]
num_labels = 131073  # Based on your specification

output_file = 'ama131k_n_3_test.txt'

with open(output_file, 'w') as f:
    # Write header
    f.write(f"{num_data_points} {num_features} {num_labels}\n")

    # Write each data point with target labels and feature values
    for i in range(num_data_points):
        # Extract target labels for the current data point
        target_indices = target_labels[i]

        # Extract non-zero feature indices and corresponding values from sparse TF-IDF matrix
        non_zero_indices = tfidf_matrix[i].nonzero()[1]  # Column indices with non-zero values
        non_zero_values = tfidf_matrix[i, non_zero_indices].toarray().flatten()  # Corresponding non-zero values

        # Create index:value pairs
        index_value_pairs = ' '.join(f"{idx}:{val:.6f}" for idx, val in zip(non_zero_indices, non_zero_values))

        # Create the target labels string
        target_pair = ','.join(map(str, target_indices))

        # Format the line and write it to the file
        line = f"{target_pair} {index_value_pairs}\n"
        f.write(line)

print(f"Data saved to {output_file}")
