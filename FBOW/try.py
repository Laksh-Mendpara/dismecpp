import numpy as np
from gensim.models import KeyedVectors
from transformers import AutoTokenizer
from nltk.corpus import stopwords
from collections import Counter
import re
import json
import pickle

# Load the pre-trained Word2Vec model
print('loading word2vec')
model_path = './GoogleNews-vectors-negative300.bin'
w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Define stop words
stop_words = set(stopwords.words('english'))

# Sample data loading (assumes you have a 'trn.json' file with data)
with open('trn.json', 'r') as file:
    data = file.readlines()

# Function to clean and preprocess the text
def preprocess_text(text):
    # Remove content inside any kind of brackets
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|\<.*?\>', '', text)
    # Remove numbers, stopwords, punctuation, and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    return tokens

# Process titles and build vocabulary with frequency filtering
vocab_counter = Counter()

c = 0
for entry in data:
    c += 1
    if c%1000 == 0: print(c)
    json_data = json.loads(entry)
    title = json_data['title']
    tokens = preprocess_text(title)
    vocab_counter.update(tokens)

# Filter vocabulary based on frequency and presence in Word2Vec model
min_frequency = 4
filtered_vocab = [token for token, freq in vocab_counter.items() if freq >= min_frequency and token in w2v_model]

# Select top l most frequent words from filtered vocab
l = 200  # or any other desired number of top words
top_words = [word for word, _ in vocab_counter.most_common(l) if word in filtered_vocab]

# Initialize matrix A with dimensions vocab_size x l
vocab_size = len(filtered_vocab)
print('vocab_size:', vocab_size)
A = np.zeros((vocab_size, l))

# Calculate cosine similarity and fuzzy membership function (Aij)
for i, word in enumerate(filtered_vocab):
    word_vector = w2v_model[word]
    for j, top_word in enumerate(top_words):
        top_word_vector = w2v_model[top_word]
        cosine_similarity = np.dot(word_vector, top_word_vector) / (np.linalg.norm(word_vector) * np.linalg.norm(top_word_vector))
        if cosine_similarity > 0:
            A[i, j] = cosine_similarity
        else:
            A[i, j] = 0


print(top_words)

# print(A)
print(A.shape)

# Save the filtered vocabulary and membership matrix
with open(f'filtered_vocab_titles_{l}.pkl', 'wb') as vocab_file:
    pickle.dump(filtered_vocab, vocab_file)

with open(f'membership_matrix_titles_{l}.npy', 'wb') as matrix_file:
    np.save(matrix_file, A)

print("Filtered vocabulary and membership matrix saved successfully.")
