import numpy as np
import tensorflow as tf
import pandas as pd
import os
import _pickle as cPickle
from matching import bidirectional_match,get_last_output
from load_data import *
import pymorphy2
import string 

list_punctuation = set(string.punctuation)
list_punctuation = list(list_punctuation) + ['"', '’', "''", '``', '`', '—', '«', '»']
morph = pymorphy2.MorphAnalyzer() # для приведения слов в начальную форму


def log(message,file_path=os.path.join('bimpm_lstm','log.txt')):

    print(message)
    f1=open(file_path, 'a+')
    f1.write(message)
    f1.write("\n")
    f1.close()

def one_hot(batch_size,Y):

    B = np.zeros((batch_size,2))

    B[np.arange(batch_size),Y] = 1

    return B.astype(int)

def length(sequence):
	
	used = tf.sign(sequence)
	length = tf.reduce_sum(used, 1)
	length = tf.cast(length, tf.int32)
	
	return length

class My_model:

    def __init__(self):

        self.word2idx = {'PAD' : 0}
        self.weights = []

        self.features = {}
        self.word_list = []

        self.emb_path = 'wordvec_rv_3.txt'
        self.dataset_path_train = 'train_PP.tsv'
        self.dataset_path_test = 'test_PP.tsv'

        self.learning_rate = 0.0001


    def load_w2v(self):

        cache_file = os.path.join('cache', 'bimpm.pkl')
        word2idx_cache_file = os.path.join('cache', 'word2idx.pkl')

        print('Creating word2vec embeddings:')

        with open(self.emb_path,'r') as file:

            for index, line in enumerate(file):
                values = line.split()
                word = values[0]
                if 1:
#                     word = morph.parse(word.lower())[0].normal_form  

                    try:
                        word_weights = np.asarray(values[1:],dtype=np.float32)
                        if(word_weights.shape[0] == 300):
                            self.word2idx[word] = index+1
                            self.weights.append(word_weights)

                    except ValueError:
                        print('Error at line ',index)

                if index == 100000:
                    break

        self.embed_dim = len(self.weights[0])
        pad_vec = np.random.randn(self.embed_dim)
        pad_vec_norm = pad_vec / np.linalg.norm(pad_vec)
        self.weights.insert(0, pad_vec_norm) #np.random.randn(self.embed_dim))

        self.word2idx['UNK'] = len(self.weights) # unknown
        self.weights.append(np.random.randn(self.embed_dim))

        self.weights = np.stack(self.weights)

        self.vocab_size = self.weights.shape[0]

        print('Saving word2idx to: ' + word2idx_cache_file)
        with open(word2idx_cache_file, 'wb') as f:
            cPickle.dump(self.word2idx,f)
        print('Done!')

        print('Saving word2vec embeddings to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            cPickle.dump(self.weights,f)
        print('Done!')


        print(self.vocab_size,self.embed_dim)


        print("Shape of word2vec embeddings:",self.weights.shape)
        # print(self.weights)
