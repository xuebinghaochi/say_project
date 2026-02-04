from xml.sax import InputSource

from data import train_data,test_data
import numpy as np

vocab=list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print('%d unique words found' % vocab_size)
word_to_idx={w:i for i,w in enumerate(vocab)}
idx_to_idx={i:w for i,w in enumerate(vocab)}
# print(word_to_idx['good'])
# print(idx_to_idx[0])

def createInputs(text):
    Input=[]
