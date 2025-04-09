from gensim.models import KeyedVectors #type:ignore
import gensim.downloader as gensim_api #type:ignore

import numpy as np #type: ignore
import pandas as pd #type: ignore
import matplotlib.pyplot as plt #type:ignore
import re
from bs4 import BeautifulSoup as bs #type:ignore


def load_word2vec(model_name = 'word2vec-google-news-300'):
    model_path = gensim_api.load(model_name, return_path= True)
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary= True)
    return word_vectors


def clean_text(text):
    text = bs(text, 'html.parser').get_text()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"#\S+|@\S+", "", text)  # Removes hashtags & mentions
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)  # Keeps only alphanumeric characters and spaces
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def text_to_tokens(text, stopwords):
    get_stop_words = set(stopwords.words('english'))
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    tokens = [token for token in tokens if token not in get_stop_words]
    return tokens

def split_data(data_path, train_size, test_size, seed):
    np.random.seed(seed)
    df = pd.read_csv(data_path)
    reviews, sentiments = df.review.values.tolist(), df.sentiment.values.tolist()
    # convert all sentiments to integers 
    for i in range(len(sentiments)):
        sentiments[i] = 0 if sentiments[i] == 'negative' else 1

    random_indices = np.arange(len(reviews))
    np.random.shuffle(random_indices)
    train_reviews, train_sentiments = [reviews[i] for i in random_indices[:train_size]], \
                                        [sentiments[i] for i in random_indices[:train_size]]
    test_reviews, test_sentiments = [reviews[i] for i in random_indices[train_size:train_size + test_size]], \
                                        [sentiments[i] for i in random_indices[train_size:train_size+test_size]]
    val_reviews, val_sentiments = [reviews[i] for i in random_indices[train_size + test_size:]], \
                                        [sentiments[i] for i in random_indices[train_size+test_size:]]
    
    sentiment_data = dict(
        train_reviews = train_reviews,
        train_sentiments = train_sentiments,
        test_reviews = test_reviews,
        test_sentiments = test_sentiments,
        val_reviews = val_reviews,
        val_sentiments = val_sentiments
    )

    return sentiment_data


def tokenize_data(document):
    '''
    document: should be a list of strings where each strings is one sample
    '''
    tokenized_document = []
    for sentence in document:
        tokenized = text_to_tokens(sentence)
        tokenized_document.append(tokenized)
    return tokenized_document
    
def get_info(x, y, data_type):
    print(f'\nDetails of {data_type} data: \n')
    assert len(x) == len(y)
    print(f'Number of Samples in {data_type}: {len(x)} ')
    print('Number of positive sentiments: ', np.sum([1 for i in y if i==1]))
    print('Number of negative sentiments: ', np.sum([1 for i in y if i==0]))


def clean_data(train, val, test):
    clean_xtrain = []
    for text in train:
        cleaned_text = clean_text(text)
        clean_xtrain.append(cleaned_text)

    clean_xval = []
    for text in val:
        cleaned_text = clean_text(text)
        clean_xval.append(cleaned_text)

    clean_xtest = []
    for text in test:
        cleaned_text = clean_text(text)
        clean_xtest.append(cleaned_text)

    return clean_xtrain, clean_xval, clean_xtest

def preprocess(datapath, train_size, test_size, print_info, seed):
    train_test_val_split = split_data(datapath, train_size, test_size, seed)

    ###---LOAD DATA IN DATA, LABEL FORMAT---###
    xtrain, ytrain = train_test_val_split['train_reviews'], train_test_val_split['train_sentiments']
    xtest, ytest = train_test_val_split['test_reviews'], train_test_val_split['test_sentiments']
    xval, yval = train_test_val_split['val_reviews'], train_test_val_split['val_sentiments']

    if print_info:
        get_info(xtrain, ytrain, 'Train')
        get_info(xtest, ytest, 'Test')
        get_info(xval, yval, 'Validation')

    xtrain, xval, xtest = clean_data(xtrain, xval, xtest)
    return [(xtrain, ytrain), (xval, yval), (xtest, ytest)]


if __name__ == '__main__':

    SEED = 746
    np.random.seed(SEED)
