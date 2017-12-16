'''
Evaluation code for the SICK dataset (SemEval 2014 Task 1)
'''

import os
import numpy as np
import os.path
import shutil
import keras.backend as K
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.utils import shuffle

from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Merge, Input, Multiply, Layer
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

class Abs(Layer):
    def __init__(self, **kwargs):
        super(Abs, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        inp1, inp2 = x[0],x[1]
        return K.abs(inp1-inp2)
    
    def get_output_shape_for(self, input_shape):
        return input_shape

def evaluate(encoder, seed=1234, evaltest=False, loc='F:\\workspace\\project\\Siamese\\skip-thoughts-master\\data\\'):
    """
    Run experiment
    """
    print 'Preparing data...'
    train, dev, test, scores = load_data(loc)
    train[0], train[1], scores[0] = shuffle(train[0], train[1], scores[0], random_state=seed)
    
    print 'Computing training skipthoughts...'
    trainA = encoder.encode(train[0], verbose=False, use_eos=True)
    trainB = encoder.encode(train[1], verbose=False, use_eos=True)
    
    print 'Computing development skipthoughts...'
    devA = encoder.encode(dev[0], verbose=False, use_eos=True)
    devB = encoder.encode(dev[1], verbose=False, use_eos=True)

    print 'Computing test skipthoughts...'
    testA = encoder.encode(test[0], verbose=False, use_eos=True)
    testB = encoder.encode(test[1], verbose=False, use_eos=True)

    print 'Computing feature combinations...'
    testF = np.c_[np.abs(testA - testB), testA * testB]

    print 'Computing feature combinations...'
    trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
    #devF = np.c_[np.abs(devA - devB), devA * devB]
    
    devF = testF
    
    #trainF = np.c_[trainA, trainB]
    #devF = np.c_[devA, devB]
    
    print 'Computing external feature ...'
    feat_train = np.load('feat_train.npy')
    feat_test = np.load('feat_test.npy')
    
    x_train = np.load('x_train.npy')
    x_test = np.load('x_test.npy')
    
    feat1_train = np.hstack((feat_train, x_train))
    feat1_test = np.hstack((feat_test, x_test))
    
    ss = StandardScaler()
    ss.fit(np.vstack((feat1_train, feat1_test)))
    feat1_train = ss.transform(feat1_train)
    feat1_test = ss.transform(feat1_test)
    
    #feat_dev = feat1_train[len(trainF):]
    feat1_train = feat1_train[0:len(trainF)]
    
    feat_dev = feat1_test
    
    print 'Encoding labels...'
    
    trainY = encode_labels(scores[0])
    #devY = encode_labels(scores[1])
    devY = encode_labels(scores[2])
    #print 'few changing....'
    #scores[0] = [i-1 for i in scores[0]]
    #scores[1] = [i-1 for i in scores[1]]
    #scores[2] = [i-1 for i in scores[2]]
    
    #trainY = to_categorical(np.round(scores[0]),5)
    #devY = to_categorical(np.round(scores[1]),5)

    print 'Compiling model...'
    lrmodel = prepare_model(ninputs=trainF.shape[1],n_feats=feat1_train.shape[1])

    print 'Training...'
    bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY, scores[2], feat1_train, feat_dev)
    #bestlrmodel = train_model(lrmodel, trainF, trainY, devF, devY, scores[1])

    if evaltest:

        print 'Evaluating...'
        r = np.arange(1,6)
        yhat = np.dot(bestlrmodel.predict([testF,feat1_test], verbose=2), r)
        #yhat = np.dot(bestlrmodel.predict_proba(testF, verbose=2), r)
        pr = pearsonr(yhat, scores[2])[0]
        sr = spearmanr(yhat, scores[2])[0]
        se = mse(yhat, scores[2])
        print 'Test Pearson: ' + str(pr)
        print 'Test Spearman: ' + str(sr)
        print 'Test MSE: ' + str(se)

        return yhat


def prepare_model(ninputs=9600,n_feats=47, nclass=5):
    """
    Set up and compile the model architecture (Logistic regression)
    """
    inp1 = Input(shape=(ninputs,))
    inp2 = Input(shape=(n_feats,))
    out_neurons1 = 50
    out_neurons2 = 20
    out_neurons2 = 10
    m1 = Dense(input_dim=ninputs, output_dim=out_neurons1,activation='sigmoid')(inp1)
    m2 = Dense(input_dim=ninputs, output_dim=out_neurons1,activation='softmax')(inp2)
    
    m1 = Dense(input_dim=ninputs, output_dim=out_neurons2,activation='sigmoid')(m1)
    m2 = Dense(input_dim=ninputs, output_dim=out_neurons2,activation='softmax')(m2)
    
    #m1 = Dense(input_dim=ninputs, output_dim=out_neurons2,activation='sigmoid')(m1)
    #m2 = Dense(input_dim=ninputs, output_dim=out_neurons2,activation='softmax')(m2)
    
    m = Merge(mode='concat')([m1,m2])
    
    #mul = Multiply()([m1,m2])
    #add = Abs()([m1,m2])
    #m = Merge(mode='concat')([mul,add])
    
    score = Dense(output_dim=nclass,activation='softmax')(m)
    model = Model([inp1,inp2],score)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
    '''
    lrmodel = Sequential()
    lrmodel.add(Dense(input_dim=ninputs, output_dim=nclass))
    #lrmodel.add(Activation('softmax'))
    #lrmodel.compile(loss='categorical_crossentropy', optimizer='adam')
    
    #return lrmodel
    
    model_feat = Sequential()
    model_feat.add(Dense(input_dim=27, output_dim=nclass))
    merge_model = Sequential()
    merge_model.add(Merge([lrmodel, model_feat], mode='concat'))
    merge_model.add(Dense(output_dim=nclass))
    merge_model.add(Activation('softmax'))
    merge_model.compile(loss='categorical_crossentropy', optimizer='adam')
    return merge_model'''
    
    '''lrmodel.add(Dense(input_dim=ninputs, output_dim=1000,activation = 'relu'))
    lrmodel.add(Dropout(0.5))
    lrmodel.add(Dense(output_dim=500,activation = 'relu'))
    lrmodel.add(Dropout(0.5))
    lrmodel.add(Dense(output_dim=nclass))'''
    #return merge_model


def train_model(lrmodel, X, Y, devX, devY, devscores, feat_train, feat_dev):
#def train_model(lrmodel, X, Y, devX, devY, devscores):

    """
    Train model, using pearsonr on dev for early stopping
    """
    done = False
    best = -1.0
    r = np.arange(1,6)
    num=0
    #print type(X)

    while not done:
        # Every 100 epochs, check Pearson on development set
        lrmodel.fit([X,feat_train], Y, verbose=2, shuffle=False, nb_epoch =2, validation_data=([devX,feat_dev], devY))
        yhat = np.dot(lrmodel.predict([devX,feat_dev] , verbose=2), r)
        #lrmodel.fit(X, Y, verbose=2, shuffle=False, validation_data=(devX, devY))
        #yhat = np.dot(lrmodel.predict_proba(devX , verbose=2), r)
        
        score = pearsonr(yhat, devscores)[0]
        
        if score > best:
            print score,num
            best = score
            #print type(X)
            bestlrmodel = prepare_model(ninputs=X.shape[1],n_feats=feat_train.shape[1])
            weights = lrmodel.get_weights()
            #print type(weights)
            bestlrmodel.set_weights(weights)
            #print 'thois coe'
            #bst_models.append(lrmodel)
            #lrmodel.save('models\\model')
            #print 'here'
            #num+=1
            
        else:
            done = True   
            
    #bestlrmodel = load_model('models\\model'+str(num))
    
    yhat = np.dot(bestlrmodel.predict([devX,feat_dev], verbose=2), r)
    #yhat = np.dot(bestlrmodel.predict_proba(devX, verbose=2), r)
    score = pearsonr(yhat, devscores)[0]
    print 'Dev Pearson: ' + str(score)
    return bestlrmodel
    

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


def load_data(loc='..\\data\\'):
    """
    Load the SICK semantic-relatedness dataset
    """
    trainA, trainB, devA, devB, testA, testB = [],[],[],[],[],[]
    trainS, devS, testS = [],[],[]

    with open(os.path.join(loc, 'SICK_train.txt'), 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            trainA.append(text[1])
            trainB.append(text[2])
            trainS.append(text[3])
    with open(os.path.join(loc, 'SICK_trial.txt'), 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            devA.append(text[1])
            devB.append(text[2])
            devS.append(text[3])
    with open(os.path.join(loc, 'SICK_test_annotated.txt'), 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            testA.append(text[1])
            testB.append(text[2])
            testS.append(text[3])

    trainS = [float(s) for s in trainS[1:]]
    devS = [float(s) for s in devS[1:]]
    testS = [float(s) for s in testS[1:]]

    return [trainA[1:], trainB[1:]], [devA[1:], devB[1:]], [testA[1:], testB[1:]], [trainS, devS, testS]

