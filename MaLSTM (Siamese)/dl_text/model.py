# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers.core import Dense, Reshape, Permute
from keras.layers import Input, merge, ZeroPadding2D,GlobalAveragePooling2D,GlobalMaxPooling1D,GlobalAveragePooling1D,ZeroPadding1D,AveragePooling1D, GlobalMaxPooling2D, Dropout, Merge, Conv1D, Lambda, Flatten,  Conv2D, MaxPooling2D, UpSampling2D, Convolution2D

from dl_text.dl import word2vec_embedding_layer

######################## MODEL USING BASIC CNN ########################

def cnn(embedding_matrix, dimx=50, dimy=50, nb_filter = 120, 
        embedding_dim = 50,filter_length = (50,4), vocab_size = 8000, depth = 1):

    print 'Model Uses Basic CNN......'
    
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    inpy = Input(shape=(dimy,),dtype='int32',name='inpy')
    
    x = word2vec_embedding_layer(embedding_matrix,train=False)(inpx)
    y = word2vec_embedding_layer(embedding_matrix,train=False)(inpy)
    
    x = Permute((2,1))(x)
    y = Permute((2,1))(y)

    conv1 = Reshape((embedding_dim,dimx,1))(x)
    conv2 = Reshape((embedding_dim,dimy,1))(y)   
       
    channel_1, channel_2 = [], []
    
    for dep in range(depth):
        
        #conv1 = ZeroPadding2D((filter_width - 1, 0))(conv1)
        #conv2 = ZeroPadding2D((filter_width - 1, 0))(conv2)
        

        ques = Conv2D(nb_filter=nb_filter, kernel_size = filter_length, activation='relu',
                data_format = 'channels_last',border_mode="valid")(conv1)
        ans = Conv2D(nb_filter, kernel_size = filter_length, activation='relu',
                data_format="channels_last",border_mode="valid")(conv2)
                    
            
        #conv1 = GlobalMaxPooling2D()(ques)
        #conv2 = GlobalMaxPooling2D()(ans)
        #conv1 = MaxPooling2D()(ques)
        #conv2 = MaxPooling2D()(ans)
        
        channel_1.append(GlobalMaxPooling2D()(ques))
        channel_2.append(GlobalMaxPooling2D()(ans))
        
        #channel_1.append(GlobalAveragePooling2D()(ques))
        #channel_2.append(GlobalAveragePooling2D()(ans))
    
    h1 = channel_1.pop(-1)
    if channel_1:
        h1 = merge([h1] + channel_1, mode="concat")

    h2 = channel_2.pop(-1)
    if channel_2:
        h2 = merge([h2] + channel_2, mode="concat")
    
    h =  Merge(mode="concat",name='h')([h1, h2])
    #h = Dropout(0.2)(h)
    #h = Dense(50, kernel_regularizer=regularizers.l2(reg2),activation='relu')(h)
    #wrap = Dropout(0.5)(h)
    #wrap = Dense(64, activation='tanh')(h)   
    
    score = Dense(2,activation='softmax',name='score')(h)
    model = Model([inpx, inpy],[score])
    model.compile( loss='categorical_crossentropy',optimizer='adam')
    
    return model

def cnn_ft(embedding_matrix, dimx=50, dimy=50, dimft=44, nb_filter = 120, 
        embedding_dim = 50,filter_length = (50,4), vocab_size = 8000, depth = 1):

    print 'Model Uses CNN with Features......'
    
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    inpy = Input(shape=(dimy,),dtype='int32',name='inpy')
    inpft = Input(shape=(dimft,),dtype='int32',name='inpft')
    
    x = word2vec_embedding_layer(embedding_matrix,train=False)(inpx)
    y = word2vec_embedding_layer(embedding_matrix,train=False)(inpy)
    
    
    x = Permute((2,1))(x)
    y = Permute((2,1))(y)

    conv1 = Reshape((embedding_dim,dimx,1))(x)
    conv2 = Reshape((embedding_dim,dimy,1))(y)   
       
    channel_1, channel_2 = [], []
    
    for dep in range(depth):
        filter_width = filter_length[1]
        conv1 = ZeroPadding2D((filter_width - 1, 0))(conv1)
        conv2 = ZeroPadding2D((filter_width - 1, 0))(conv2)
        

        ques = Conv2D(nb_filter=nb_filter, kernel_size = filter_length, activation='relu',
                data_format = 'channels_last',border_mode="valid")(conv1)
        ans = Conv2D(nb_filter, kernel_size = filter_length, activation='relu',
                data_format="channels_last",border_mode="valid")(conv2)
                    
            
        #conv1 = GlobalMaxPooling2D()(ques)
        #conv2 = GlobalMaxPooling2D()(ans)
        #conv1 = MaxPooling2D()(ques)
        #conv2 = MaxPooling2D()(ans)
        
        channel_1.append(GlobalMaxPooling2D()(ques))
        channel_2.append(GlobalMaxPooling2D()(ans))
        
        #channel_1.append(GlobalAveragePooling2D()(ques))
        #channel_2.append(GlobalAveragePooling2D()(ans))
    
    h1 = channel_1.pop(-1)
    if channel_1:
        h1 = merge([h1] + channel_1, mode="concat")

    h2 = channel_2.pop(-1)
    if channel_2:
        h2 = merge([h2] + channel_2, mode="concat")
    
    h =  Merge(mode="concat",name='h')([h1, h2, inpft])
    #h = Dropout(0.2)(h)
    #h = Dense(50, kernel_regularizer=regularizers.l2(reg2),activation='relu')(h)
    #wrap = Dropout(0.5)(h)
    #wrap = Dense(64, activation='tanh')(h)   
    
    score = Dense(2,activation='softmax',name='score')(h)
    model = Model([inpx, inpy, inpft],[score])
    model.compile( loss='categorical_crossentropy',optimizer='adam')
    
    return model
