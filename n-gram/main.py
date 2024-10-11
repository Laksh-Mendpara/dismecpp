import json
import spacy
import re
from collections import Counter
from nltk.corpus import stopwords
from itertools import islice
import pickle

# Load SpaCy's English model
nlp = spacy.load('en_core_web_sm')

# Load the stop words
stop_words = set(stopwords.words('english'))

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

# Function to generate n-grams from tokens
def generate_ngrams(tokens, n):
    ngrams = zip(*[islice(tokens, i, None) for i in range(n)])
    return ['_'.join(ngram) for ngram in ngrams]

# Load the JSON data and process each title
with open('trn.json', 'r') as f:
    c = 0
    for line in f:
        c += 1
        if c % 500 == 0:
            print(c)
        data_point = json.loads(line)
        title = data_point.get('title', '')
        cleaned_tokens = clean_title(title)

        # Generate uni-grams, bi-grams, tri-grams, etc. (adjust n as needed)
        for n in range(1, 4):  # n=1 for unigrams, n=2 for bigrams, n=3 for trigrams
            ngrams = generate_ngrams(cleaned_tokens, n)
            vocab_counter.update(ngrams)

# Filter vocabulary to include only tokens/grams with frequency > 5 and limit the size of the vocabulary
max_vocab_size = 300000  # Adjust vocab size as needed
filtered_vocabulary = [word for word, freq in vocab_counter.most_common(max_vocab_size) if freq > 2]

# Output the filtered vocabulary
print(len(filtered_vocabulary))
print(filtered_vocabulary[:20])
print(filtered_vocabulary[-20: -1])

# Save the filtered vocabulary to a .pkl file
with open('vocabulary_large.pkl', 'wb') as vocab_file:
    pickle.dump(filtered_vocabulary, vocab_file)

# Process some example titles and display the resulting n-grams
with open('trn.json', 'r') as f:
    c = 0
    for line in f:
        if c > 20:
            break
        c += 1
        print(c)
        data_point = json.loads(line)
        title = data_point.get('title', '')
        cleaned_tokens = clean_title(title)

        # Generate n-grams for the title and filter based on the vocabulary
        final_tokens = []
        for n in range(1, 4):
            ngrams = generate_ngrams(cleaned_tokens, n)
            final_tokens.extend([token for token in ngrams if token in filtered_vocabulary])

        print(title)
        print(final_tokens)
        print('----------')
