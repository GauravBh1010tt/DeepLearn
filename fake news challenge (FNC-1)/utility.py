# -*- coding: utf-8 -*-

"""
Created on Tue Mar 07 11:48:18 2017

@author: Gaurav
"""

import math
import random
import pickle
import warnings
import nltk
import itertools
import numpy as np
import keras.backend as K
from theano import tensor as T
import gensim as gen
import scipy.stats as measures
from gensim.models import word2vec
from scipy.stats.stats import pearsonr
from keras.engine.topology import Layer
from nltk.tokenize import regexp_tokenize
from sklearn.metrics.pairwise import cosine_similarity

warnings.simplefilter("ignore")

START = '$_START_$'
END = '$_END_$'
unk_token = '$_UNK_$'
#vocab_size = 17000
#embedding_dim = 300
#dimx = 30
#dimy = 30
loss_type = 2 # 1 - l1+l2+l3-L4; 2 - l2+l3-L4; 3 - l1+l2+l3 , 4 - l2+l3
#word_to_index={}
#index_to_word=[]


#wordVec_model = word2vec.Word2Vec.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz",binary=True)
    
def split(train_l,train_r,label,ratio):
    total = train_l.shape[0]
    train_samples = int(total*(1-ratio))
    test_samples = total-train_samples
    tr_l,tst_l,tr_r,tst_r,l_tr,l_tst=[],[],[],[],[],[]
    dat=random.sample(range(total),train_samples)
    for a in dat:
        tr_l.append(train_l[a,:])
        tr_r.append(train_r[a,:])
        l_tr.append(label[a])
        
    for i in range(test_samples):
        if i not in dat:
            tst_l.append(train_l[i,:])
            tst_r.append(train_r[i,:])
            l_tst.append(label[i])
            
    tr_l = np.array(tr_l)
    tr_r = np.array(tr_r)
    tst_l = np.array(tst_l)
    tst_r = np.array(tst_r)
    l_tr = np.array(l_tr)
    l_tst = np.array(l_tst)
    
    return tr_l,tst_l,tr_r,tst_r,l_tr,l_tst
    
class ZeroLike(Layer):
    def __init__(self, **kwargs):
        super(ZeroLike, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.zeros_like(x)

    def get_output_shape_for(self, input_shape):
        return input_shape

def project(model,inp):
    m = model.predict([inp[0],inp[1]])
    return m[2]
    
def sum_corr(view1,view2,flag=''):
    
    print("test correlation")
    corr = 0
    for i,j in zip(view1,view2):
        corr += measures.pearsonr(i,j)[0]
    print('avg sum corr ::',flag,'::',corr/len(view1))

def cal_sim(model,ind1,ind2=1999):
    view1 = np.load("test_v1.npy")[0:ind1]
    view2 = np.load("test_v2.npy")[0:ind2]
    label1 = np.load('test_l.npy')
    x1 = project(model,[view1,np.zeros_like(view1)])
    x2 = project(model,[np.zeros_like(view2),view2])
    label2 = []
    count = 0
    MAP=0
    for i,j in enumerate(x1):
        cor = []
        AP=0
        for y in x2:
            temp1 = j.tolist()
            temp2 = y.tolist()
            cor.append(pearsonr(temp1,temp2))
        #if i == np.argmax(cor):
        #    count+=1
        #val=[(q,(i*ind1+p))for p,q in enumerate(cor)]
        val=[(q,p)for p,q in enumerate(cor)]
        val.sort()
        val.reverse()
        label2.append(val[0:4])
        t = [w[1]for w in val[0:7]]
        #print t
        for x,y in enumerate(t):
            if y in range(i,i+5):
                AP+=1/(x+1)
        print(t)
        print(AP)
        MAP+=AP
    #print 'accuracy  :- ',float(count)*100/ind1,'%'
    print('MAP is : ',MAP/ind1)

def cos_sim(ind1,ind2=1999):
    view1 = np.load("test_v1.npy")[0:ind1]
    view2 = np.load("test_v2.npy")[0:ind2]
    #val = []
    MAP=0
    for i,j in enumerate(view1):
        val=[]
        AP=0
        for x in view2:            
            val.append(cosine_similarity(j,x)[0].tolist())
        #val=val[0].tolist()
        #print val[0].tolist()
        val=[(q,p)for p,q in enumerate(val)]
        #print val
        val.sort()
        val.reverse()
        t = [w[1]for w in val[0:7]]
        for x,y in enumerate(t):
            if y in range(i,i+5):
                AP+=1/(x+1)
        print(t)
        print(AP)
        MAP+=AP
    print('MAP is : ',MAP/ind1)

class sample:
    def __init__(self):
        print("Inside utility")
    def process_data(self,sent_Q,sent_A,wordVec_model=None,dimx=100,dimy=100,vocab_size=10000,embedding_dim=300):
    #if True:
        sent1 = []
        #sent1_Q = ques_sent
        #sent1_A = ans_sent
        sent1.extend(sent_Q)
        #sent.extend(ques_sent)
        sent1.extend(sent_A)
    #    sent1 = [' '.join(i) for i in sent1]
        #sent.extend(ans_sent)
        sentence = ["%s %s %s" % (START,x,END) for x in sent1]
        self.tokenize_sent = [regexp_tokenize(x, 
                                         pattern = '\w+|$[\d\.]+|\S+') for x in sentence]
                        
        freq = nltk.FreqDist(itertools.chain(*self.tokenize_sent))
        print('found ',len(freq),' unique words')
        vocab = freq.most_common(vocab_size - 1)
        self.index_to_word = [x[0] for x in vocab]
        self.index_to_word.append(unk_token)
        #for i in index_to_word1:
        #    index_to_word.append(i)
        
        self.word_to_index = dict([(w,i) for i,w in enumerate(self.index_to_word)])
        
        # for key in word_to_index1.keys():
        #    word_to_index[key] = word_to_index1[key]
        
        for i,sent in enumerate(self.tokenize_sent):
            self.tokenize_sent[i] = [w if w in self.word_to_index else unk_token for w in sent]
        
        self.len_train = len(sent_Q)
        text=[]
        for i in self.tokenize_sent:
            text.extend(i)
        
        self.sentences_x = []
        self.sentences_y = []
        
        #print 'here' 
        
        for sent in self.tokenize_sent[0:self.len_train]:
            temp = [START for i in range(dimx)]
            for ind,word in enumerate(sent[0:dimx]):
                temp[ind] = word
            self.sentences_x.append(temp)
            
        for sent in self.tokenize_sent[self.len_train:]:
            temp = [START for i in range(dimy)]
            for ind,word in enumerate(sent[0:dimy]):
                temp[ind] = word       
            self.sentences_y.append(temp)
            
        X_data = []
        for i in self.sentences_x:
            temp = []
            for j in i:
                temp.append(self.word_to_index[j])
            temp = np.array(temp).T
            X_data.append(temp)
        
        y_data=[]
        for i in self.sentences_y:
            temp = []
            for j in i:
                temp.append(self.word_to_index[j])
            temp = np.array(temp).T
            y_data.append(temp)
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        #model = gen.models.Word2Vec.load('Word2Vec_QA')  
    
        embedding_matrix = np.zeros((len(self.index_to_word) + 1,embedding_dim))
        
        unk = []
        for i,j in enumerate(self.index_to_word):
            try:
                embedding_matrix[i] = wordVec_model[j]
            except:
                #embedding_matrix[i] = np.zeros_like()
                
                unk.append(j)
                continue
        print('number of unkown words: ',len(unk))
        print('some unknown words ',unk[0:5])
        return X_data,y_data,embedding_matrix