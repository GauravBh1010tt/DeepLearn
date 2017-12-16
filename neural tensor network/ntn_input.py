import scipy.io as sio
import numpy as np
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import gensim as gen
import keras.backend as K
from keras import optimizers
from theano import tensor as T
from keras.models import Model
import scipy.stats as measures
from keras.layers import Convolution1D, Conv2D, MaxPooling2D, Flatten
from gensim.models import word2vec
from keras.engine.topology import Layer
from sklearn.metrics.pairwise import cosine_similarity
from keras import initializers, regularizers, constraints 
from keras.layers import Input, Merge,Lambda, Embedding, Bidirectional, LSTM, Dense, RepeatVector, Dropout

embedding_size = 100

data_number = 0 #0 - Wordnet, 1 - Freebase
if data_number == 0: data_name = 'Wordnet'
else: data_name = 'Freebase'

data_path = 'data\\'+data_name
output_path = 'data\\output\\'+data_name+'\\'

entities_string='/entities.txt'
relations_string='/relations.txt'
embeds_string='/initEmbed.mat'
training_string='/train.txt'
test_string='/test.txt'
dev_string='/dev.txt'

#input: path of dataset to be used
#output: python list of entities in dataset
def load_entities(data_path=data_path):
    entities_file = open(data_path+entities_string)
    entities_list = entities_file.read().strip().split('\n')
    entities_file.close()
    return entities_list

#input: path of dataset to be used
#output: python list of relations in dataset
def load_relations(data_path=data_path):
    relations_file = open(data_path+relations_string)
    relations_list = relations_file.read().strip().split('\n')
    relations_file.close()
    return relations_list
    

#input: path of dataset to be used
#output: python dict from entity string->1x100 vector embedding of entity as precalculated
def load_init_embeds(data_path=data_path):
    embeds_path = data_path+embeds_string
    return load_embeds(embeds_path)

#input: Generic function to load embeddings from a .mat file
def load_embeds(file_path):
    mat_contents = sio.loadmat(file_path)
    words = mat_contents['words']
    we = mat_contents['We']
    tree = mat_contents['tree']
    word_vecs = [[we[j][i] for j in range(embedding_size)] for i in range(len(words[0]))]
    entity_words = [map(int, tree[i][0][0][0][0][0]) for i in range(len(tree))]
    return (word_vecs,entity_words)

def load_training_data(data_path=data_path):
    training_file = open(data_path+training_string)
    training_data = [line.split('\t') for line in training_file.read().strip().split('\n')]
    return np.array(training_data)

def load_dev_data(data_path=data_path):
    #print data_path+dev_string
    dev_file = open(data_path+dev_string)
    dev_data = [line.split('\t') for line in dev_file.read().strip().split('\n')]
    return np.array(dev_data)

def load_test_data(data_path=data_path):
    test_file = open(data_path+test_string)
    test_data = [line.split('\t') for line in test_file.read().strip().split('\n')]
    return np.array(test_data)
