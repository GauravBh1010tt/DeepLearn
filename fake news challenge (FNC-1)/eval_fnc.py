'''
Evaluation code for the SICK dataset (SemEval 2014 Task 1)
'''
import numpy as np
import os.path
from util import *
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.utils import shuffle
from utils.score import report_score
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Embedding
from keras import regularizers
from keras.layers import Merge, Input, Multiply, Layer
from sklearn.preprocessing import StandardScaler


def split(train_l,train_r,label,ratio):
    total = train_l.shape[0]
    train_samples = int(total*(1-ratio))
    test_samples = total-train_samples
    tr_l,tst_l,tr_r,tst_r,l_tr,l_tst=[],[],[],[],[],[]
    dat=random.sample(range(total),train_samples)
    for a in dat:
        tr_l.append(train_l[a])
        tr_r.append(train_r[a])
        l_tr.append(label[a])
    print 'splitting - validation samples ',test_samples
    for i in range(total):
        if i not in dat:
            tst_l.append(train_l[i])
            tst_r.append(train_r[i])
            l_tst.append(label[i])
    print 'splitting - train samples ',len(dat)        
    tr_l = np.array(tr_l)
    tr_r = np.array(tr_r)
    tst_l = np.array(tst_l)
    tst_r = np.array(tst_r)
    l_tr = np.array(l_tr)
    l_tst = np.array(l_tst)
    
    return tr_l,tst_l,tr_r,tst_r,l_tr,l_tst

def load_dataset(file_trhead, file_trbody, file_tshead, file_tsbody):
    trhead = pd.read_csv(file_trhead)
    trbody = pd.read_csv(file_trbody)
    tshead = pd.read_csv(file_tshead)
    tsbody = pd.read_csv(file_tsbody)
    tr_head_array = trhead.values
    tr_body_array = trbody.values                    
    ts_head_array = tshead.values
    ts_body_array = tsbody.values
    tr_labels = tr_head_array[:,2]
    ts_labels = ts_head_array[:,2]
    
    tr_body_id = tr_head_array[:,1]
    train_dh = tr_head_array[:,0]         ##########
    train_db = []
    for i in range(len(tr_head_array)):
        for j in range(len(tr_body_array)):
            if tr_body_array[j][0] == tr_body_id[i]:
                train_db.append(tr_body_array[j][1])
                break
    tr_lab = []
    for i in tr_labels:
        if i == 'unrelated':
            tr_lab.append(3)
        if i == 'agree':
            tr_lab.append(0)
        if i == 'discuss':
            tr_lab.append(2)
        if i == 'disagree':
            tr_lab.append(1)
    train_db = np.array(train_db)               ##############
    
    
    ts_body_id = ts_head_array[:,1]
    test_dh = ts_head_array[:,0]         ##########
    test_db = []
    for i in range(len(ts_head_array)):
        for j in range(len(ts_body_array)):
            if ts_body_array[j][0] == ts_body_id[i]:
                test_db.append(ts_body_array[j][1])
                break
    ts_lab = []
    for i in ts_labels:
        if i == 'unrelated':
            ts_lab.append(3)
        if i == 'agree':
            ts_lab.append(0)
        if i == 'discuss':
            ts_lab.append(2)
        if i == 'disagree':
            ts_lab.append(1)
    
    test_db= np.array(test_db)                  #############
    
    
    #signs=['?','.',]
    print("Refining train datset")
    train_rdh = []
    for i in range(len(train_dh)):
        sentence = ""
        for char in train_dh[i]:
            if char.isalpha() or char == ' ':
                sentence+=char.lower()
            else:
                sentence+=' '
        train_rdh.append(sentence)
    
    train_rdb = []
    for i in range(len(train_db)):
        sentence = ""
        for char in train_db[i]:
            if char.isalpha() or char == ' ':
                sentence+=char.lower()
            else:
                sentence+=' '
        train_rdb.append(sentence)
    
    print("Refining test datset")
    test_rdh = []
    for i in range(len(test_dh)):
        sentence = ""
        for char in test_dh[i]:
            if char.isalpha() or char == ' ':
                sentence+=char.lower()
            else:
                sentence+=' '
        test_rdh.append(sentence)
    
    test_rdb = []
    for i in range(len(test_db)):
        sentence = ""
        for char in test_db[i]:
            if char.isalpha() or char == ' ':
                sentence+=char.lower()
            else:
                sentence+=' '
        test_rdb.append(sentence)
    
    dic = pd.read_pickle('stop_dic')
    
    train_new_rdb = []
    test_new_rdb = []

    word_limit = 250    
    print 'removing stop words and using', word_limit,'words limit .....'
    
    for i in train_rdb:
        temp=[]
        for j in i.split():
            try:
                a=dic[j]
            except:
                temp.append(j)
        train_new_rdb.append(' '.join(temp[0:min(len(temp),word_limit)]))
        
    for i in test_rdb:
        temp=[]
        for j in i.split():
            try:
                a=dic[j]
            except:
                temp.append(j)
        test_new_rdb.append(' '.join(temp[0:min(len(temp),word_limit)]))
    
    train_rdh = np.array(train_rdh)
    test_rdh = np.array(test_rdh)
    train_new_rdb = np.array(train_new_rdb)
    test_new_rdb = np.array(test_new_rdb)
    
    
    return train_rdh, train_new_rdb, test_rdh, test_new_rdb
    #tr_h, dev_h, tr_b, dev_b, tr_s, dev_s = split(np.array(train_rdh), np.array(train_rdb), tr_lab, 0.2)
    #return [tr_h, tr_b], [dev_h, dev_b], [tr_s, dev_s]
    
def evaluate(encoder=None, seed=1234, evaltest=False, loc='./data/'):
    """
    Run experiment
    """
    print 'Preparing data for fnc...'
    
    #train, dev, test, scores = load_data(loc)
    #train[0], train[1], scores[0] = shuffle(train[0], train[1], scores[0], random_state=seed)

    '''
    trh, trb, tsh, tsb =\
                load_dataset("/fnc_data/train_stances.csv", "/fnc_data/train_bodies.csv",\
                             "/fnc_data/competition_test_stances.csv", "/fnc_data/test_bodies.csv")
   '''
    train_h = np.load('/fncdata2/encode_train_head.npy')
    train_b = np.load('/fncdata2/encode_train_body.npy')
    test_h = np.load('/fncdata2/encode_test_head.npy')
    test_b = np.load('/fncdata2/encode_test_body.npy')
    score_train = np.load('/fncdata2/score_train.npy')
    score_test = np.load('/fncdata2/score_test.npy')
    #train_b = big_mat
    #train_h, dev_h, train_b, dev_b, score_train, dev_score = split(np.array(train_h), train_b, score_train, 0.2)
 
    print 'loading training skipthoughts...'
    #trainA = encoder.encode(train_h, verbose=False, use_eos=True)
    #trainB = encoder.encode(train_b, verbose=False, use_eos=True)
    trainA = train_h
    trainB = train_b
    
    print 'Computing development skipthoughts...'
    #devA = encoder.encode(dev_h, verbose=False, use_eos=True)
    #devB = encoder.encode(dev_b, verbose=False, use_eos=True)
#    devA = dev_h
#    devB = dev_b
    devA = test_h
    devB = test_b
    dev_score = score_test

    print 'Computing feature combinations...'
    trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
    devF = np.c_[np.abs(devA - devB), devA * devB]

    print 'Encoding labels...'
    #trainY = encode_labels(train_labels)
    #devY = encode_labels(holdout_labels)
    trainY = to_categorical(score_train, 4)
    devY = to_categorical(dev_score, 4)
    
    train_Fx, test_Fx = load_features()
    #fmodel = generate_feature_model(train_Fx, score_train, test_Fx, dev_score, ninputs=len(train_Fx[0]))

    train_tfidf, test_tfidf = generate_tfidf()
    
    print 'Compiling model...'
    lrmodel = prepare_model(ninputs=trainF.shape[1],n_feats=train_Fx.shape[1],n_tfidf=train_tfidf.shape[1])

    print 'Training...'
    bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY, dev_score, train_Fx, test_Fx, train_tfidf, test_tfidf)
    
    if evaltest:
        print 'Loading test skipthoughts...'
        testA = test_h
        testB = test_b

        print 'Computing feature combinations...'
        testF = np.c_[np.abs(testA - testB), testA * testB]
        
        yhat = bestlrmodel.predict(testF, verbose=2)
        yhat = [i.argmax()for i in yhat]
        
        string_predicted,test_stances = [],[]
        
        for i,j in zip(yhat,score_test):
            if i == 3:
                string_predicted.append('unrelated')
            if i == 0:
                string_predicted.append('agree')
            if i == 2:
                string_predicted.append('discuss')
            if i == 1:
                string_predicted.append('disagree')
            if j == 3:
                test_stances.append('unrelated')
            if j == 0:
                test_stances.append('agree')
            if j == 2:
                test_stances.append('discuss')
            if j == 1:
                test_stances.append('disagree')
                
        report_score(test_stances,string_predicted)
        score = accuracy_score(score_test, yhat)
        print 'accuracy is ..',score
        #print 'Evaluating...'


def generate_tfidf():
    file_train_instances = "/fncdata/train_stances.csv"
    file_train_bodies = "/fncdata/train_bodies.csv"
    file_test_instances = "/fncdata/competition_test_stances.csv"
    file_test_bodies = "/fncdata/test_bodies.csv"
    raw_train = FNCData(file_train_instances, file_train_bodies)
    raw_test = FNCData(file_test_instances, file_test_bodies)
    n_train = len(raw_train.instances)

    lim_unigram = 5000
    train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
    pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
    #feature_size = len(train_set[0])
    test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    return np.array(train_set), np.array(test_set)

def prepare_model(ninputs=9600, n_feats=45,nclass=4,n_tfidf=10001):
    inp1 = Input(shape=(ninputs,))
    inp2 = Input(shape=(n_feats,))
    inp3 = Input(shape=(n_tfidf,))
    reg = 0.00005
    out_neurons1 = 500
    #out_neurons2 = 20
    #out_neurons2 = 10
    m1 = Dense(input_dim=ninputs, output_dim=out_neurons1,activation='sigmoid'\
                      ,kernel_regularizer=regularizers.l2(0.00000001))(inp1)
    m1 = Dropout(0.2)(m1)
    m1 = Dense(100,activation='sigmoid')(m1)
    #m1 = Dropout(0.2)(m1)
    #m1 = Dense(4, activation='sigmoid')(m1)
    
    #m2 = Dense(input_dim=n_feats, output_dim=n_feats,activation='relu')(inp2)
    m2 = Dense(50,activation='relu')(inp2)
    #m2=Dense(4,activation='relu')(m2)
    
    m3 = Dense(500, input_dim=n_tfidf, activation='relu',\
                    kernel_regularizer=regularizers.l2(reg))(inp3)
    
    m3 = Dropout(0.4)(m3)
    m3 = Dense(50, activation='relu')(m3)
    #m3 = Dropout(0.4)(m3)
    #m3 = Dense(4, activation='softmax')(m3)
    
    
    #m1 = Dense(input_dim=ninputs, output_dim=out_neurons2,activation='sigmoid')(m1)
    #m2 = Dense(input_dim=ninputs, output_dim=out_neurons2,activation='softmax')(m2)
    
    m = Merge(mode='concat')([m1,m2,m3])
    
    #mul = Multiply()([m1,m2])
    #add = Abs()([m1,m2])
    #m = Merge(mode='concat')([mul,add])
    
    score = Dense(output_dim=nclass,activation='softmax')(m)
    model = Model([inp1,inp2,inp3],score)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def prepare_model2(ninputs=9600, n_feats=45,nclass=4,n_tfidf=10001):
    inp1 = Input(shape=(ninputs,))
    inp2 = Input(shape=(n_feats,))
    inp3 = Input(shape=(n_tfidf,))
    reg = 0.00005
    out_neurons1 = 500
    #out_neurons2 = 20
    #out_neurons2 = 10
    m1 = Dense(input_dim=ninputs, output_dim=out_neurons1,activation='sigmoid'\
                      ,kernel_regularizer=regularizers.l2(0.00000001))(inp1)
    m1 = Dropout(0.2)(m1)
    m1 = Dense(100,activation='sigmoid')(m1)
    #m1 = Dropout(0.2)(m1)
    #m1 = Dense(4, activation='sigmoid')(m1)
    
    m2 = Dense(input_dim=n_feats, output_dim=n_feats,activation='relu')(inp2)
    m2 = Dense(4,activation='relu')(inp2)
    #m2=Dense(4,activation='relu')(m2)
    
    m3 = Dense(500, input_dim=n_tfidf, activation='relu',\
                    kernel_regularizer=regularizers.l2(reg))(inp3)
    
    m3 = Dropout(0.4)(m3)
    m3 = Dense(50, activation='relu')(m3)
    #m3 = Dropout(0.4)(m3)
    #m3 = Dense(4, activation='softmax')(m3)
    
    
    #m1 = Dense(input_dim=ninputs, output_dim=out_neurons2,activation='sigmoid')(m1)
    #m2 = Dense(input_dim=ninputs, output_dim=out_neurons2,activation='softmax')(m2)
    
    m = Merge(mode='concat')([m1,m2,m3])
    
    #mul = Multiply()([m1,m2])
    #add = Abs()([m1,m2])
    #m = Merge(mode='concat')([mul,add])
    
    score = Dense(output_dim=nclass,activation='softmax')(m)
    model = Model([inp1,inp2,inp3],score)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def prepare_model1(ninputs=9600, n_feats=45,nclass=4,n_tfidf=10001):
    inp1 = Input(shape=(ninputs,))
    inp2 = Input(shape=(n_feats,))
    inp3 = Input(shape=(n_tfidf,))
    reg = 0.00005
    out_neurons1 = 500
    #out_neurons2 = 20
    #out_neurons2 = 10
    m1 = Dense(input_dim=ninputs, output_dim=out_neurons1,activation='sigmoid'\
                      ,kernel_regularizer=regularizers.l2(0.00000001))(inp1)
    m1 = Dropout(0.5)(m1)
    m1 = Dense(100,activation='sigmoid')(m1)
    m1 = Dropout(0.5)(m1)
    
    m2 = Dense(input_dim=n_feats, output_dim=n_feats,activation='relu')(inp2)
    m2 = Dense(30,activation='relu')(m2)
    
    
    m3 = Dense(500, input_dim=n_tfidf, activation='relu',\
                    kernel_regularizer=regularizers.l2(reg))(inp3)
    
    m3 = Dropout(0.6)(m3)
    m3 = Dense(100, activation='relu')(m3)
    m3 = Dropout(0.4)(m3)
    m3 = Dense(4, activation='softmax')(m3)
    
    
    #m1 = Dense(input_dim=ninputs, output_dim=out_neurons2,activation='sigmoid')(m1)
    #m2 = Dense(input_dim=ninputs, output_dim=out_neurons2,activation='softmax')(m2)
    
    m = Merge(mode='concat')([m1,m2,m3])
    
    #mul = Multiply()([m1,m2])
    #add = Abs()([m1,m2])
    #m = Merge(mode='concat')([mul,add])
    
    score = Dense(output_dim=nclass,activation='softmax')(m)
    model = Model([inp1,inp2,inp3],score)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
    
    """
    Set up and compile the model architecture (Logistic regression)
    
    print 'changed'
    out_neurons1 = 500
    lrmodel = Sequential()
    lrmodel.add(Dense(input_dim=ninputs, output_dim=out_neurons1,activation='sigmoid'\
                      ,kernel_regularizer=regularizers.l2(0.00000001)))
    lrmodel.add(Dropout(0.5))
    #lrmodel.add(Dense(out_neurons2))
    #lrmodel.add(Dropout(0.5))
    lrmodel.add(Dense(output_dim=nclass))
    
    #lrmodel.add(Dense(input_dim=ninputs, output_dim=nclass))
    #lrmodel.add(Dropout(0.3))
    lrmodel.add(Activation('softmax'))
    lrmodel.compile(loss='categorical_crossentropy', optimizer='adam')
    return lrmodel
    """

def train_model(lrmodel, X, Y, devX, devY, devscores, feat_train, feat_dev, train_tfidf, test_tfidf):
    """
    Train model, using pearsonr on dev for early stopping
    """
    done = False
    best = -1.0
    #r = np.arange(1,5)
    
    while not done:
        # Every 100 epochs, check Pearson on development set
        lrmodel.fit([X,feat_train,train_tfidf], Y, verbose=2, shuffle=False, nb_epoch = 3, validation_data=([devX,feat_dev,test_tfidf], devY))
        #yhat = np.dot(lrmodel.predict(devX, verbose=2), r)
        yhat = lrmodel.predict([devX,feat_dev,test_tfidf], verbose=2)
        yhat = [i.argmax()for i in yhat]
        
        string_predicted,test_stances = [],[]
    
        for i,j in zip(yhat,devscores):
            if i == 3:
                string_predicted.append('unrelated')
            if i == 0:
                string_predicted.append('agree')
            if i == 2:
                string_predicted.append('discuss')
            if i == 1:
                string_predicted.append('disagree')
            if j == 3:
                test_stances.append('unrelated')
            if j == 0:
                test_stances.append('agree')
            if j == 2:
                test_stances.append('discuss')
            if j == 1:
                test_stances.append('disagree')
        print 'using new limit value....'
        #score = accuracy_score(devscores, yhat)
        score = report_score(test_stances,string_predicted,val=True)
        #return lrmodel
    
        if score > best:
            print score
            best = score
            bestlrmodel = prepare_model(ninputs=X.shape[1],n_feats=feat_train.shape[1],n_tfidf=train_tfidf.shape[1])
            bestlrmodel.set_weights(lrmodel.get_weights())
        else:
            done = True
            print '***** best model obtained with score',best,'******'

    yhat = bestlrmodel.predict([devX, feat_dev, test_tfidf], verbose=2)
    yhat = [i.argmax()for i in yhat]
    string_predicted,test_stances = [],[]
    
    for i,j in zip(yhat,devscores):
        if i == 3:
            string_predicted.append('unrelated')
        if i == 0:
            string_predicted.append('agree')
        if i == 2:
            string_predicted.append('discuss')
        if i == 1:
            string_predicted.append('disagree')
        if j == 3:
            test_stances.append('unrelated')
        if j == 0:
            test_stances.append('agree')
        if j == 2:
            test_stances.append('discuss')
        if j == 1:
            test_stances.append('disagree')
            
    report_score(test_stances,string_predicted)
    return bestlrmodel

import math

def load_features():
    
    train_hand = np.load('/fncdata3/hand.train.npy')
    #train_overlap = np.load('/fncdata3/overlap.train.npy')
    #train_refuting = np.load('/fncdata3/refuting.train.npy')
    #train_polarity = np.load('/fncdata3/polarity.train.npy')
    test_hand = np.load('/fncdata3/hand.test.npy')
    #test_overlap = np.load('/fncdata3/overlap.test.npy')
    #test_refuting = np.load('/fncdata3/refuting.test.npy')
    #test_polarity = np.load('/fncdata3/polarity.test.npy')
    '''
    train_other = np.load('/fncdata4/x_train.npy')
    test_other = np.load('/fncdata4/x_test.npy')
    train_other = train_other[:,16]
    test_other = test_other[:,16]
    #train_X = np.c_[train_polarity, train_refuting, train_overlap]
    #test_X = np.c_[test_polarity, test_refuting, test_overlap]
    for k,i in enumerate(test_other):
        if math.isnan(i):
            #print 'here',k
            test_other[k] = 0.0
    
    train_X = np.c_[train_hand, train_other]
    test_X = np.c_[test_hand, test_other]
    
    train_feat = np.load('/fncdata3/feat_train.npy')
    train_other = np.load('/fncdata3/x_train.npy')
    test_feat = np.load('/fncdata3/feat_test.npy')
    test_other = np.load('/fncdata3/x_test.npy')
    train_X = np.c_[train_feat, train_other]
    test_X = np.c_[test_feat, test_other]
    
    for k,i in enumerate(test_X):
        for ind,j in enumerate(i):
            if math.isnan(j):
                #print 'here',k
                test_X[k][ind] = 0.0
        
    ss = StandardScaler()
    ss.fit(np.vstack((train_X, test_X)))
    feat1_train = ss.transform(train_X)
    feat1_test = ss.transform(test_X)
    
    #feat_dev = feat1_train[len(trainF):]
    #feat1_train = feat1_train[0:len(trainF)]
    
    #feat_dev = feat1_test
    '''
    return train_hand, test_hand