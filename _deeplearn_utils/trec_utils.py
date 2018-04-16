
"""
** deeplean-ai.com **
** dl-lab **
created by :: GauravBh1010tt
"""

import numpy as np

from dl_text import *
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical

################### LOADING, CLEANING AND PROCESSING DATASET ###################
def load_trec(model_name, glove_fname):
    
    train_file = open('../_deeplearn_utils/data/trec/train-filtered.tsv')
    test_file= open('../_deeplearn_utils/data/trec/test-filtered.tsv')
    dev_file = open('../_deeplearn_utils/data/trec/dev-filtered.tsv')
    res_fname = 'test.ref'
    pred_fname = 'pred_%s'%model_name

    train_header = train_file.readline().split('\t')
    test_header = test_file.readline().split('\t')
    dev_header = dev_file.readline().split('\t')
    
    data_train = train_file.readlines()
    data_test= test_file.readlines()
    data_dev = dev_file.readlines()
    
    data_train = [line.split('\t') for line in data_train]
    data_test = [line.split('\t') for line in data_test]
    data_dev = [line.split('\t') for line in data_dev]
        
    ques_train, ans_train, label_train = [], [], []
    ques_test, ans_test, label_test = [], [], []
    ques_dev, ans_dev, label_dev = [], [], []
        
    for line in data_train:
        ques_train.append(dl.clean(line[1]))
        ans_train.append(dl.clean(line[5]))
        label_train.append(int(line[-1][0]))
        
    for line in data_test:
        ques_test.append(dl.clean(line[1]))
        ans_test.append(dl.clean(line[5]))
        label_test.append(int(line[-1][0]))
            
    for line in data_dev:
        ques_dev.append(dl.clean(line[1]))
        ans_dev.append(dl.clean(line[5]))
        label_dev.append(int(line[-1][0]))
    
    ques, ans = [], []
    
    for i in [ques_train, ques_test, ques_dev]:
        ques.extend(i)
    
    for i in [ans_train, ans_test, ans_dev]:
        ans.extend(i)
    
    train_len = len(data_train)
    test_len = len(data_test)
    
    wordVec_model = dl.loadGloveModel(glove_fname)
    
    feat_LS = np.load('../_deeplearn_utils/Extracted_Features/trec/lex.npy')
    feat_read = np.load('../_deeplearn_utils/Extracted_Features/trec/read.npy')
    feat_numeric = np.load('../_deeplearn_utils/Extracted_Features/trec/numeric.npy')
    
    feat = np.hstack((feat_LS, feat_read, feat_numeric))
    
    feat_train = feat[:train_len]
    feat_test = feat[train_len:(test_len + train_len)]
    #
    ss = StandardScaler()
    ss.fit(feat)
    feat_train = ss.transform(feat_train)
    feat_test = ss.transform(feat_test)
    
    return ques, ans, to_categorical(label_train), train_len, test_len, wordVec_model, res_fname, pred_fname, feat_train, feat_test


def prepare_train_test(data_l,data_r,train_len,test_len):
    
    X_train_l = data_l[:train_len]
    X_test_l = data_l[train_len:(test_len + train_len)]
    X_dev_l = data_l[(test_len + train_len):]
    
    X_train_r = data_r[:train_len]
    X_test_r = data_r[train_len:(test_len + train_len)]
    X_dev_r = data_r[(test_len + train_len):]
    
    return X_train_l,X_test_l,X_dev_l,X_train_r,X_test_r,X_dev_r
