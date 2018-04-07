"""
** deeplean-ai.com **
** dl-lab **
created by :: GauravBh1010tt
"""
from __future__ import print_function

from keras import backend as K
from keras.models import Model
from keras import regularizers
from keras.engine.topology import Layer
from keras.layers.core import Dense, Reshape, Permute
from keras.layers import Input, Embedding, GlobalAveragePooling2D, GlobalMaxPooling2D,GlobalMaxPooling1D, Bidirectional, Dense, Dropout, Merge, Multiply, Conv1D, Lambda, Flatten, LSTM, TimeDistributed, Conv2D, MaxPooling2D, UpSampling2D

from dl_text.dl import word2vec_embedding_layer
from dl_layers.layers import Similarity

def cnn_sim(embedding_matrix, dimx=50, dimy=50, nb_filter = 120, 
        embedding_dim = 50,filter_length = (50,4), vocab_size = 8000, depth = 1):

    print('Model Uses CNN with Sim......')
    
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    inpy = Input(shape=(dimy,),dtype='int32',name='inpy')
    
    x = word2vec_embedding_layer(embedding_matrix,train=True)(inpx)
    y = word2vec_embedding_layer(embedding_matrix,train=True)(inpy)
    
    x = Permute((2,1))(x)
    y = Permute((2,1))(y)

    conv1 = Reshape((embedding_dim,dimx,1))(x)
    conv2 = Reshape((embedding_dim,dimy,1))(y)   
       
    channel_1, channel_2 = [], []
    
    for dep in range(depth):
        
        #conv1 = ZeroPadding2D((filter_width - 1, 0))(conv1)
        #conv2 = ZeroPadding2D((filter_width - 1, 0))(conv2)
        

        ques = Conv2D(nb_filter=nb_filter, kernel_size = filter_length, activation='relu',
                data_format = 'channels_last')(conv1)
        ans = Conv2D(nb_filter, kernel_size = filter_length, activation='relu',
                data_format="channels_last")(conv2)
                    
        ques = Dropout(0.5)(ques)
        ans = Dropout(0.5)(ans)
            
        ques = GlobalMaxPooling2D()(ques)        
        ans = GlobalMaxPooling2D()(ans)
        
        ques = Dropout(0.5)(ques)
        ans = Dropout(0.5)(ans)
        
        channel_1.append(ques)
        channel_2.append(ans)
        
        #channel_1.append(GlobalAveragePooling2D()(ques))
        #channel_2.append(GlobalAveragePooling2D()(ans))
    
    h1 = channel_1.pop(-1)
    if channel_1:
        h1 = merge([h1] + channel_1, mode="concat")

    h2 = channel_2.pop(-1)
    if channel_2:
        h2 = merge([h2] + channel_2, mode="concat")
    
    sim = Similarity(nb_filter)([h1,h2])
    h =  Merge(mode="concat",name='h')([h1, sim, h2])
    #h = Dropout(0.2)(h)
    #h = Dense(50, kernel_regularizer=regularizers.l2(reg2),activation='relu')(h)
    #wrap = Dropout(0.5)(h)
    #wrap = Dense(64, activation='tanh')(h)   
    
    score = Dense(2,activation='softmax',name='score')(h)
    model = Model([inpx, inpy],[score])
    model.compile( loss='categorical_crossentropy',optimizer='adam')
    
    return model

def cnn_sim_ft(embedding_matrix, dimx=50, dimy=50, dimft=44, nb_filter = 120, 
        embedding_dim = 50,filter_length = (50,4), vocab_size = 8000, depth = 1):

    print('Model Uses CNN with Sim and Features......')
    
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    inpy = Input(shape=(dimy,),dtype='int32',name='inpy')
    inpft = Input(shape=(dimft,),dtype='int32',name='inpft')
    
    x = word2vec_embedding_layer(embedding_matrix,train=True)(inpx)
    y = word2vec_embedding_layer(embedding_matrix,train=True)(inpy)
    
    
    x = Permute((2,1))(x)
    y = Permute((2,1))(y)

    conv1 = Reshape((embedding_dim,dimx,1))(x)
    conv2 = Reshape((embedding_dim,dimy,1))(y)   
       
    channel_1, channel_2 = [], []
    
    for dep in range(depth):
        #filter_width = filter_length[1]
        #conv1 = ZeroPadding2D((filter_width - 1, 0))(conv1)
        #conv2 = ZeroPadding2D((filter_width - 1, 0))(conv2)
        

        ques = Conv2D(nb_filter=nb_filter, kernel_size = filter_length, activation='relu',
                data_format = 'channels_last',border_mode="valid")(conv1)
        ans = Conv2D(nb_filter, kernel_size = filter_length, activation='relu',
                data_format="channels_last",border_mode="valid")(conv2)
                    
            
        ques = Dropout(0.5)(ques)
        ans = Dropout(0.5)(ans)
            
        ques = GlobalMaxPooling2D()(ques)        
        ans = GlobalMaxPooling2D()(ans)
        
        ques = Dropout(0.5)(ques)
        ans = Dropout(0.5)(ans)
        
        channel_1.append(ques)
        channel_2.append(ans)
        
        #channel_1.append(GlobalAveragePooling2D()(ques))
        #channel_2.append(GlobalAveragePooling2D()(ans))
    
    h1 = channel_1.pop(-1)
    if channel_1:
        h1 = merge([h1] + channel_1, mode="concat")

    h2 = channel_2.pop(-1)
    if channel_2:
        h2 = merge([h2] + channel_2, mode="concat")
    
    sim = Similarity(nb_filter)([h1,h2])
    h =  Merge(mode="concat",name='h')([h1, sim, h2, inpft])
    #h = Dropout(0.2)(h)
    #h = Dense(50, kernel_regularizer=regularizers.l2(reg2),activation='relu')(h)
    #wrap = Dropout(0.5)(h)
    #wrap = Dense(64, activation='tanh')(h)   
    
    score = Dense(2,activation='softmax',name='score')(h)
    model = Model([inpx, inpy, inpft],[score])
    model.compile( loss='categorical_crossentropy',optimizer='adam')
    
    return model
