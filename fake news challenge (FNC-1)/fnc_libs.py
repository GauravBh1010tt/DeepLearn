# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
import pandas as pd
from keras.layers.convolutional import Convolution1D
import utility
import warnings
from nltk.tokenize import regexp_tokenize
import numpy as np
import gensim as gen
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Layer,Lambda, Dropout, Activation, Input, Merge, Multiply
from keras.layers import Embedding, Bidirectional, LSTM, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission


dimx = 50
dimy = 250
vocab_size = 15000

def trainCNN(obj, wordVec_model, dataset_headLines, dataset_body):
    embedding_dim = 300
    LSTM_neurons = 50
    dense_neuron = 16
    lamda = 0.0
    nb_filter = 100
    filter_length = 4
    batch_size = 50
    epochs = 5
    ntn_out = 16
    ntn_in = nb_filter 
    state = False
    
    train_head,train_body,embedding_matrix = obj.process_data(sent_Q=dataset_headLines,
                                                     sent_A=dataset_body,dimx=dimx,dimy=dimy,
                                                     wordVec_model = wordVec_model)    
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')
    #x = Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=dimx)(inpx)
    x = word2vec_embedding_layer(embedding_matrix)(inpx)  
    inpy = Input(shape=(dimy,),dtype='int32',name='inpy')
    #y = Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=dimy)(inpy)
    y = word2vec_embedding_layer(embedding_matrix)(inpy)
    
    
    ques = Convolution1D(nb_filter=nb_filter, filter_length=filter_length,
                         border_mode='valid', activation='relu',
                         subsample_length=1)(x)
                            
    ans = Convolution1D(nb_filter=nb_filter, filter_length=filter_length,
                        border_mode='valid', activation='relu',
                        subsample_length=1)(y)
            
    hx = Lambda(max_1d, output_shape=(nb_filter,))(ques)
    hy = Lambda(max_1d, output_shape=(nb_filter,))(ans)
    #hx = GlobalMaxPooling1D()(ques)
    #hy = GlobalMaxPooling1D()(ans)
    
    #hx = Flatten()
    #hy = Flatten()
    
    hx1 = Multiply()([hx,hy])
    hy1 = Abs()([hx,hy])
    h =  Merge(mode="concat",name='h')([hx1,hy1])
    
    '''
    shared_lstm = Bidirectional(LSTM(LSTM_neurons,return_sequences=True),merge_mode='sum')   
    #shared_lstm = LSTM(LSTM_neurons,return_sequences=True)    
    hx = shared_lstm(x)
    #hx = Dropout(0.2)(hx)
    hy = shared_lstm(y)
    #hy = Dropout(0.2)(hy)
    
    #corr = CorrelationRegularization(-lamda)([hx,hy])     
   
    h1 = Flatten()(hx)
    h2 = Flatten()(hy)
    hx1 = Multiply()([h1,h2])
    hx2 = Abs()([h1,h2])
    h =  Merge(mode="concat",name='h')([hx1,hx2])
    '''
    #h1 = Multiply()([hx,hy])
    #h2 = Abs()([hx,hy])
    #h =  Merge(mode="concat",name='h')([h1,h2])

    #h =  Merge(mode="concat",name='h')([hx,hy])
    #h = NeuralTensorLayer(output_dim=1,input_dim=ntn_in)([hx,hy])
    #h = ntn_layer(ntn_in,ntn_out,activation=None)([hx,hy])
    #score = h
    wrap = Dense(dense_neuron, activation='relu',name='wrap')(h)
    #score = Dense(1,activation='sigmoid',name='score')(h)
    #wrap = Dense(dense_neuron,activation='relu',name='wrap')(h)
    score = Dense(4,activation='softmax',name='score')(wrap)
    
    #score=K.clip(score,1e-7,1.0-1e-7)
    #corr = CorrelationRegularization(-lamda)([hx,hy])
    #model = Model( [inpx,inpy],[score,corr])
    model = Model( [inpx,inpy],score)
    model.compile( loss='categorical_crossentropy',optimizer="adadelta",metrics=['accuracy'])    
    return model,train_head,train_body

class Abs(Layer):
    def __init__(self, **kwargs):
        super(Abs, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.abs(x[0]- x[1])

    def get_output_shape_for(self, input_shape):
        return input_shape

def max_1d(X):
    return K.max(X, axis=1)

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y


def applyKFold(folds, hold_out, fold_stances, hold_out_stances):
    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf

    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]
    report_score(actual,predicted)


def word2vec_embedding_layer(embedding_matrix):
    #weights = np.load('Word2Vec_QA.syn0.npy')
    layer = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix])
    return layer    

def generateMatrix(obj, sent_Q, sent_A):
    START = '$_START_$'
    END = '$_END_$'
    unk_token = '$_UNK_$'
    #dimx = 100
    #dimy = 200
    sent1 = []
    #sent1_Q = ques_sent
    #sent1_A = ans_sent
    sent1.extend(sent_Q)
    #sent.extend(ques_sent)
    sent1.extend(sent_A)
    #sent1 = [' '.join(i) for i in sent1]
    #sent.extend(ans_sent)
    sentence = ["%s %s %s" % (START,x,END) for x in sent1]
    tokenize_sent = [regexp_tokenize(x, 
                                     pattern = '\w+|$[\d\.]+|\S+') for x in sentence]
                        
    #for i in index_to_word1:
    #    index_to_word.append(i)
    # for key in word_to_index1.keys():
    #    word_to_index[key] = word_to_index1[key]
        
    for i,sent in enumerate(tokenize_sent):
        tokenize_sent[i] = [w if w in obj.word_to_index else unk_token for w in sent]
        
    len_train = len(sent_Q)
    text=[]
    for i in tokenize_sent:
        text.extend(i)
        
    sentences_x = []
    sentences_y = []
        
        #print 'here' 
        
    for sent in tokenize_sent[0:len_train]:
        temp = [START for i in range(dimx)]
        for ind,word in enumerate(sent[0:dimx]):
            temp[ind] = word
        sentences_x.append(temp)

    for sent in tokenize_sent[len_train:]:
        temp = [START for i in range(dimy)]
        for ind,word in enumerate(sent[0:dimy]):
            temp[ind] = word       
        sentences_y.append(temp)
        
    X_data = []
    for i in sentences_x:
        temp = []
        for j in i:
            temp.append(obj.word_to_index[j])
        temp = np.array(temp).T
        X_data.append(temp)
        
    y_data=[]
    for i in sentences_y:
        temp = []
        for j in i:
            temp.append(obj.word_to_index[j])
        temp = np.array(temp).T
        y_data.append(temp)
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    return X_data, y_data

def load_data(file_head,file_body):
    head = pd.read_csv(file_head)
    body = pd.read_csv(file_body)
    head_array = head.values
    body_array = body.values                    ##########
    print(('number of headlines : ',len(head_array)))
    print(('number of news body : ',len(body_array)))
    labels = head_array[:,2]
    body_id = head_array[:,1]
    dataset_headLines = head_array[:,0]         ##########
    body_ds = []
    for i in range(len(head_array)):
        for j in range(len(body_array)):
            if body_array[j][0] == body_id[i]:
                body_ds.append(body_array[j][1])
                break
    dataset_body = np.array(body_ds)
    return  dataset_headLines, dataset_body,labels
