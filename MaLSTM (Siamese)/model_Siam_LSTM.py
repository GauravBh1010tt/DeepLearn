# -*- coding: utf-8 -*-
"""
** deeplean-ai.com **
** dl-lab **
created by :: GauravBh1010tt
"""

from keras import optimizers
from keras.models import Model
from keras.layers import Input, Flatten, Merge, Embedding, Multiply, Bidirectional, LSTM, Dense, RepeatVector, Dropout,  TimeDistributed, Lambda

from dl_text.dl import word2vec_embedding_layer
from dl_layers.layers import Abs, Exp

def S_LSTM(dimx = 30, dimy = 30, embedding_matrix=None, LSTM_neurons = 32):
    
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')
    x = word2vec_embedding_layer(embedding_matrix,train='False')(inpx)  
    inpy = Input(shape=(dimy,),dtype='int32',name='inpy')
    y = word2vec_embedding_layer(embedding_matrix,train='False')(inpy)    
    
    #hx = LSTM(LSTM_neurons)(x)
    #hy = LSTM(LSTM_neurons)(y)
   
    shared_lstm = Bidirectional(LSTM(LSTM_neurons,return_sequences=False),merge_mode='sum')   
    #shared_lstm = LSTM(LSTM_neurons,return_sequences=True)    
    hx = shared_lstm(x)
    #hx = Dropout(0.2)(hx)
    hy = shared_lstm(y)
    #hy = Dropout(0.2)(hy)
    
    h1,h2=hx,hy

    corr1 = Exp()([h1,h2])
    adadelta = optimizers.Adadelta()
    
    model = Model( [inpx,inpy],corr1)
    model.compile( loss='binary_crossentropy',optimizer=adadelta)
    
    return model
