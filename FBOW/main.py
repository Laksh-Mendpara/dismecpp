import json
import spacy
import re
from collections import Counter
from nltk.corpus import stopwords

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


# Load the JSON data and process each title
with open('trn.json', 'r') as f:
    c = 0
    for line in f:
        c+=1
        if c%500 == 0:
            print(c)
        data_point = json.loads(line)
        title = data_point.get('content', '')
        cleaned_tokens = clean_title(title)
        vocab_counter.update(cleaned_tokens)

# Filter vocabulary to include only tokens with frequency > 5
filtered_vocabulary = [word for word, freq in vocab_counter.items() if freq > 5]

# Output the filtered vocabulary
print(filtered_vocabulary)

with open('trn.json', 'r') as f:
    c = 0
    for line in f:
        if c>20:
            break
        c+=1
        print(c)
        data_point = json.loads(line)
        title = data_point.get('content', '')
        cleaned_tokens = clean_title(title)
        lis = [token for token in cleaned_tokens if token in filtered_vocabulary]
        print (title)
        print (lis)
        print('----------')
