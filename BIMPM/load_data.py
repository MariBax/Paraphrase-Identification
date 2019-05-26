import os
import nltk
import numpy as np
import tensorflow as tf
import pandas as pd
import pymorphy2
import string 
from text_processing_rv import process_custom 

list_punctuation = set(string.punctuation)
list_punctuation = list(list_punctuation) + ['"', '’', "''", '``', '`', '—', '«', '»']
morph = pymorphy2.MorphAnalyzer() # для приведения слов в начальную форму


def load_dataset(dataset_path_train,word2idx, model, process_pipeline):

    with open(dataset_path_train,'r', encoding = 'utf-8') as file1:

        sentence_one = []
        sentence_two = []
        y_true = []
        max1 = 0
        max2 = 0
        i = 0
        for line in file1:

    # if index == 0:
    # continue
            s_1 = []
            s_2 = []
            values = line.split("\t")

            #Sentence 1
#             words = values[3].split(" ")
#             words = nltk.word_tokenize(values[2])
#             words = [morph.parse(word.lower())[0].normal_form for word in words if word not in list_punctuation]
            words = process_custom(values[2],model, process_pipeline)

            s_1.extend([word2idx.get(word,word2idx['UNK']) for word in words])

            #Sentence 2
#             words = values[4].split(" ")
#             words = nltk.word_tokenize(values[3])
#             words = [word.lower() for word in words]
            words = process_custom(values[3],model, process_pipeline)

            s_2.extend([word2idx.get(word,word2idx['UNK']) for word in words])

            if len(s_1) > max1:
                max1 = len(s_1)

            if len(s_2) > max2:
                max2 = len(s_2)

            y_true.append(np.asarray(values[1]))

            sentence_one.append(np.pad(s_1,(0,26-len(s_1)),'constant',constant_values=(0)))
            sentence_two.append(np.pad(s_2,(0,26-len(s_2)),'constant',constant_values=(0)))

            i += 1

    sentence_one = np.stack(sentence_one)
    sentence_two = np.stack(sentence_two)
    y_true = np.stack(y_true)

    print("Max_train:",max1,max2)

    print(sentence_one.shape,sentence_two.shape,y_true.shape)

    return sentence_one,sentence_two,y_true


def load_dataset_test(dataset_path_test,word2idx, model, process_pipeline):

    with open(dataset_path_test,'r', encoding = 'utf-8') as file1:

        sentence_one_test = []
        sentence_two_test = []
        y_true_test = []
        max1 = 0
        max2 = 0
        index = 1
        i = 0
        for line in file1:		
    # 			if index == 0:
    # 				continue
            s_1 = []
            s_2 = []

            values = line.split("\t")

            #Sentence 1
#             words = values[3].split(" ")
#             words = nltk.word_tokenize(values[2])
#             words = [morph.parse(word.lower())[0].normal_form for word in words if word not in list_punctuation]
            words = process_custom(values[2],model, process_pipeline)
            s_1.extend([word2idx.get(word,word2idx['UNK']) for word in words])

            #Sentence 2
#             words = values[4].split(" ")
#             words = nltk.word_tokenize(values[3])
#             words = [word.lower() for word in words]
            words = process_custom(values[3],model, process_pipeline)
            s_2.extend([word2idx.get(word,word2idx['UNK']) for word in words])

            if len(s_1) > max1:
                max1 = len(s_1)

            if len(s_2) > max2:
                max2 = len(s_2)


            y_true_test.append(np.asarray(values[1]))

            sentence_one_test.append(np.pad(s_1,(0,26-len(s_1)),'constant',constant_values=(0)))
            sentence_two_test.append(np.pad(s_2,(0,26-len(s_2)),'constant',constant_values=(0)))

            i += 1

    print("Max_test:",max1,max2)

    sentence_one_test = np.stack(sentence_one_test)
    sentence_two_test = np.stack(sentence_two_test)
    y_true_test = np.stack(y_true_test)
    print(sentence_one_test.shape,sentence_two_test.shape,y_true_test.shape)

    return sentence_one_test,sentence_two_test,y_true_test
