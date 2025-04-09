import numpy as np
import re
import os
from bs4 import BeautifulSoup as bs #type:ignore
import torch
import nltk #type:ignore
from torch.utils.data import Dataset, DataLoader


def create_vocabulary(text, cutoff_freq, word_vectors):
    word_list = []

    # Process each document in texts here and store the list of all words
    for x in text:
        filtered_tokens = nltk.word_tokenize(x, language= 'english')
        word_list += filtered_tokens

    unique_words, counts = np.unique(word_list, return_counts=True)
    vocab_dict = {word: count for word, count in zip(unique_words, counts) if (count >= cutoff_freq and word_vectors.__contains__(word))}

    # Update the key values to be indices rather than counts
    for counter, key in enumerate(vocab_dict):
        vocab_dict[key] = counter + 2

    vocab_dict = dict({'<PAD>':0, '<UNK>': 1}, **vocab_dict)

    return vocab_dict

def vocab_word2vec(vocab, word_vectors):
    # Get word2vec embedding matrics for words in vocab
    vocab_w2v = []
    dummy_linear = torch.nn.Linear(2, 300)
    for key in vocab:
        if key in ['<PAD>', '<UNK>']:
            vocab_w2v.append(dummy_linear.weight.data[:, vocab[key]])
        else:
            vocab_w2v.append(torch.tensor(word_vectors[key]))

    vocab_w2v = torch.stack(vocab_w2v, dim = 0)
    return vocab_w2v

# %%
class TokenizerDataset(Dataset):
    def __init__(self, text, labels, vocab, max_seq_len = None):
        self.text = text
        self.labels = labels
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        from nltk.corpus import stopwords #type:ignore

    def encode(self, xbatch):
        '''
        xbatch: should be list of list of tokens
        '''
        encoded_xbatch = []
        for entry in xbatch:
            tokenized_entry = nltk.word_tokenize(entry, language = 'english')
            encoded_xbatch.append([self.vocab.get(token, self.vocab['<UNK>']) for token in tokenized_entry])
        
        return encoded_xbatch

    def pad(self, encoded_xbatch):
        if self.max_seq_len is None:
            max_len = max([len(l) for l in encoded_xbatch])
        else:
            max_len = self.max_seq_len

        for i, entry in enumerate(encoded_xbatch):
            if len(entry) <= max_len:
                tokens = [*entry]
                tokens += ([self.vocab['<PAD>']]*(max_len - len(entry)))
            else:
                tokens = entry[:max_len]
            encoded_xbatch[i] = tokens
        return encoded_xbatch

    def collate_fn(self, batch):
        '''
        data: should be list of cleaned strings
        '''
        max_length = 0

        xbatch = [data[0] for data in batch]
        ybatch = [data[1] for data in batch]

        encoded_xbatch = self.encode(xbatch)
        encoded_xbatch = self.pad(encoded_xbatch)
        return torch.tensor(encoded_xbatch, dtype = torch.long), torch.tensor(ybatch)
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        return self.text[idx], float(self.labels[idx])
    

# %%
# AUDIO DATASET
class AudioDataset(Dataset):
    def __init__(self, audiopath, df, target_to_idx, load_audio, win_shift):
        self.audiopath = audiopath
        self.df = df
        self.target_to_idx = target_to_idx
        self.load_audio = load_audio
        self.win_shift = win_shift

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        audio_file = self.df['filename'].iloc[index]
        path_to_file = os.path.join(self.audiopath, audio_file)
        x = self.load_audio(path_to_file, self.win_shift)
        x = np.expand_dims(x, axis = 0)
        y = self.df['target'].iloc[index]
        y = self.target_to_idx[y]
        return x, y
        

    

# %%
def create_dataloader(dataset, batch_size, shuffle = True, collate_fn = None):
    return DataLoader(dataset, batch_size= batch_size, shuffle = shuffle, collate_fn= collate_fn)
