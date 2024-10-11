import json
import spacy
import re
import numpy as np
from collections import Counter
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from textblob import TextBlob

# Load SpaCy's English model
nlp = spacy.load('en_core_web_sm')

# Load stop words
stop_words = set(stopwords.words('english'))

# Load Word2Vec model
word2vec_model_path = './GoogleNews-vectors-negative300.bin'  # Replace with your model path
print('loading word2vec model')
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

# Initialize a counter for the vocabulary
vocab_counter = Counter()


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


# Load the JSON data and process each title
with open('trn.json', 'r') as f:
    c = 0
    for line in f:
        if c>2000:
            break
        if c % 100 == 0:
            print(c)
        c += 1
        data_point = json.loads(line)
        title = data_point.get('content', '')
        cleaned_tokens = clean_title(title)
        vocab_counter.update(cleaned_tokens)

# Filter vocabulary to include only tokens with frequency > 5
filtered_vocab = [word for word, freq in vocab_counter.items() if freq > 5]

# Select top `l` words based on frequency
# top_l = 2000  # Define `l` here
top_l=10
top_words = [word for word, _ in vocab_counter.most_common(top_l)]

# Create a matrix to store dot products
vocab_size = len(vocab_counter)
print('vocab_size:', vocab_size)
top_word_vectors = []
word_vectors = []

# Create a mapping from words to index
word_to_index = {word: idx for idx, word in enumerate(vocab_counter.keys())}

# Prepare matrices for top words and all words
print('obtaining embeddings for top_l words')
for word in top_words:
    if word in word2vec_model:
        top_word_vectors.append(word2vec_model[word])
    else:
        textBlb = TextBlob(word)
        print(word, "---->", textBlb.correct().string)


print('obtaining embeddings for vocab  words')
for word in vocab_counter.keys():
    if word in word2vec_model:
        word_vectors.append(word2vec_model[word])
    else:
        textBlb = TextBlob(word)
        print(word, "---->", textBlb.correct().string)

top_word_vectors = np.array(top_word_vectors)
word_vectors = np.array(word_vectors)

# Compute dot products using matrix multiplication
print('computing dot product')
dot_product_matrix = np.dot(top_word_vectors, word_vectors.T)
print(dot_product_matrix.shape)

# # Save vocabulary
# vocab_file = 'vocabulary.json'
# with open(vocab_file, 'w') as f:
#     json.dump(filtered_vocab, f)
#
# # Save dot product matrix
# matrix_file = 'dot_product_matrix.npy'
# np.save(matrix_file, dot_product_matrix)
#
# print(f"Vocabulary saved to {vocab_file}")
# print(f"Dot product matrix saved to {matrix_file}")
