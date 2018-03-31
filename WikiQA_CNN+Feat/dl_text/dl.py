"""
** deeplean-ai.com **
** dl-lab **
created by :: GauravBh1010tt
"""

import re
import numpy as np

from nltk import FreqDist
from itertools import chain
from nltk.tokenize import regexp_tokenize

from keras import backend as K
from keras.layers import Layer, Embedding

START = '$_START_$'
END = '$_END_$'
unk_token = '$_UNK_$'

def clean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def process_data(sent_l,sent_r=None,wordVec_model=None,dimx=100,dimy=100,vocab_size=10000,embedding_dim=300):
    sent1 = []
    sent1.extend(sent_l)
    if sent_r:
        sent1.extend(sent_r)
#    sent1 = [' '.join(i) for i in sent1]
    sentence = ["%s %s %s" % (START,x,END) for x in sent1]
    tokenize_sent = [regexp_tokenize(x, 
                                     pattern = '\w+|$[\d\.]+|\S+') for x in sentence]
    
    
    freq = FreqDist(chain(*tokenize_sent))
    print 'found ',len(freq),' unique words'
    vocab = freq.most_common(vocab_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unk_token)
    
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    
    for i,sent in enumerate(tokenize_sent):
        tokenize_sent[i] = [w if w in word_to_index else unk_token for w in sent]
    
    len_train = len(sent_l)
    text=[]
    for i in tokenize_sent:
        text.extend(i)
    
    sentences_x = []
    sentences_y = []
    
    for sent in tokenize_sent[0:len_train]:
        temp = [START for i in range(dimx)]
        for ind,word in enumerate(sent[0:dimx]):
            temp[ind] = word
        sentences_x.append(temp)
            
    X_data = []
    for i in sentences_x:
        temp = []
        for j in i:
            temp.append(word_to_index[j])
        temp = np.array(temp).T
        X_data.append(temp)
    
    X_data = np.array(X_data)
    
    if sent_r:
        for sent in tokenize_sent[len_train:]:
            temp = [START for i in range(dimy)]
            for ind,word in enumerate(sent[0:dimy]):
                temp[ind] = word       
            sentences_y.append(temp)

        y_data=[]
        for i in sentences_y:
            temp = []
            for j in i:
                temp.append(word_to_index[j])
            temp = np.array(temp).T
            y_data.append(temp)
        
        y_data = np.array(y_data)

    if wordVec_model:
        embedding_matrix = np.zeros((len(index_to_word) + 1,embedding_dim))
        
        unk = []
        for i,j in enumerate(index_to_word):
            try:
                embedding_matrix[i] = wordVec_model[j]
            except:
                #print j
                unk.append(j)
                continue
        print 'number of unkown words: ',len(unk)
        print 'some unknown words ',unk[0:5]
    
    
    
    if sent_r and wordVec_model:
        return X_data,y_data,embedding_matrix
    elif sent_r:
        return X_data,y_data
    elif wordVec_model:
        return X_data,embedding_matrix
    else:
        return X_data

def prepare_train_test(data_l,data_r,train_len,test_len):
    
    X_train_l = data_l[:train_len]
    X_test_l = data_l[train_len:(test_len + train_len)]
    X_dev_l = data_l[(test_len + train_len):]
    
    X_train_r = data_r[:train_len]
    X_test_r = data_r[train_len:(test_len + train_len)]
    X_dev_r = data_r[(test_len + train_len):]
    
    return X_train_l,X_test_l,X_dev_l,X_train_r,X_test_r,X_dev_r

def word2vec_embedding_layer(embedding_matrix,train=False):
    layer = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix],trainable=train)
    return layer

def loadGloveModel(glovefile):
    print 'Loading Glove File.....'
    f = open(glovefile)
    model = {}
    for line in f:
        splitline = line.split()
        word = splitline[0]
        embedding = np.array([float(val) for val in splitline[1:]])
        model[word] = embedding
    print 'Loaded Word2Vec GloVe Model.....'
    print len(model), ' words loaded.....'
    return model

def encode_labels(labels, nclass=5):
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    Y = np.zeros((len(labels), nclass)).astype('float32')
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i+1 == np.floor(y) + 1:
                Y[j,i] = y - np.floor(y)
            if i+1 == np.floor(y):
                Y[j,i] = np.floor(y) - y + 1
    return Y