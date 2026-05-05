import re
import random
import pandas as pd
from nltk.corpus import stopwords
import nltk

REPLACE_NO_SPACE = re.compile("[._;:!*`¦\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
nltk.download('stopwords')


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = REPLACE_NO_SPACE.sub("", text)
    text = REPLACE_WITH_SPACE.sub(" ", text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    words = text.split()
    return [w for w in words if w not in stop_words]

def load_training_set(percentage_positives, percentage_negatives):
    vocab = set()
    positive_instances = []
    negative_instances = []

    df = pd.read_csv('train-positive.csv')
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_positives:
            continue
        contents = preprocess_text(contents)
        positive_instances.append(contents)
        vocab = vocab.union(set(contents))

    df = pd.read_csv('train-negative.csv')
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_negatives:
            continue
        contents = preprocess_text(contents)
        negative_instances.append(contents)
        vocab = vocab.union(set(contents))

    return positive_instances, negative_instances, vocab

def load_test_set(percentage_positives, percentage_negatives):
    positive_instances = []
    negative_instances = []

    df = pd.read_csv('test-positive.csv')
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_positives:
            continue
        contents = preprocess_text(contents)
        positive_instances.append(contents)
    df = pd.read_csv('test-negative.csv')
    
    for _, contents in df.iterrows():
        contents = contents['reviewText']
        if random.random() > percentage_negatives:
            continue
        contents = preprocess_text(contents)
        negative_instances.append(contents)

    return positive_instances, negative_instances
