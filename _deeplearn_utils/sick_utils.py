# -*- coding: utf-8 -*-
"""
** deeplean-ai.com **
** dl-lab **
created by :: GauravBh1010tt
"""
import pandas as pd
import numpy as np
import gensim as gen
from keras.utils.np_utils import to_categorical

def load_sick(model_name, wordVec=None):
    
    pred_fname = 'pred_%s'%model_name
    
    data = pd.read_csv('../_deeplearn_utils/data/sick/train_features.csv').values
    ques_sent,ans_sent,q_test,a_test = [],[],[],[]
    for i in data:
        ques_sent.append(i[0].split())
        ans_sent.append(i[1].split())
        
    data = pd.read_csv('../_deeplearn_utils/data/sick/test_features.csv').values
    for i in data:
        q_test.append(i[0].split())
        a_test.append(i[1].split())
    #data=[],[],[]
    
    score = np.array(pd.read_pickle("../_deeplearn_utils/data/sick/train_labels.pkl")).tolist()
    score = [i[0]for i in score]
    
    sc_test = np.array(pd.read_pickle("../_deeplearn_utils/data/sick/test_labels.pkl")).tolist()
    sc_test = [i[0]for i in sc_test]

    train_len = len(ques_sent)
    test_len = len(q_test)
    ques_sent.extend(q_test)
    ans_sent.extend(a_test)
    ques_sent = [' '.join(i) for i in ques_sent]
    ans_sent = [' '.join(i) for i in ans_sent]
    
    #score.extend(sc_test)
    
    train_score = score
    test_score = sc_test
    
    if wordVec !=None:
        wordVec_model = gen.models.KeyedVectors.load_word2vec_format(wordVec,binary=True)   
        return ques_sent,ans_sent, train_len, test_len, train_score, test_score, wordVec_model, pred_fname
    else:
        return ques_sent, ans_sent, train_len, test_len, train_score, test_score,  pred_fname
