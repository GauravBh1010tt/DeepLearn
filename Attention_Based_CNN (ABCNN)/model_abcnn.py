"""
** deeplean-ai.com **
** dl-lab **
created by :: GauravBh1010tt
"""

import keras
from keras.models import Model
from keras import backend as K
from keras.layers.core import Dense, Reshape, Permute
from keras.layers import Input, merge, ZeroPadding2D, RepeatVector,GlobalAveragePooling2D,GlobalMaxPooling1D,GlobalAveragePooling1D,ZeroPadding1D,AveragePooling1D, GlobalMaxPooling2D, Dropout, Merge, Conv1D, Lambda, Flatten,  Conv2D, MaxPooling2D, UpSampling2D, Convolution2D

from dl_text.dl import word2vec_embedding_layer

######################## MODEL USING BASIC CNN ########################

def abcnn(embedding_matrix, attention=1, dimx=50, dimy=50, nb_filter = 72,
          filter_length = (50,4), dropout = None, shared = 1, embedding_dim = 50, depth = 1,
          filter_widths = [4,3,2], opt_params = [0.0008,'adam']):

#if True:
    print '\n Model Uses ABCNN architecture ......'
    print 'attention : ', attention
    print 'nb_filters :', nb_filter
    print 'filter_size :', filter_length
    print 'opt params :',opt_params
    #print 'dense layer :',dense_neuron,' ',reg1
    if dropout:
        print 'using dropout'
    if shared:
        print 'using shared params'
    print '\n'
    
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    inpy = Input(shape=(dimy,),dtype='int32',name='inpy')
    
    x = word2vec_embedding_layer(embedding_matrix,train=False)(inpx)
    y = word2vec_embedding_layer(embedding_matrix,train=False)(inpy)
    
    #x = Permute((2,1))(x)
    #y = Permute((2,1))(y)
    
    mul = MatchScore(x,y)
    mulT = Permute((2,1))(mul)

    d1 = Dense(units = embedding_dim)(mul)
    d2 = Dense(units = embedding_dim)(mulT)
    
    x = Permute((2,1))(x)
    y = Permute((2,1))(y)
    
    if attention in [1,3]:

        x = Reshape(( embedding_dim, dimx, 1))(x)
        y = Reshape(( embedding_dim, dimy, 1))(y)
        d1 = Reshape((embedding_dim, dimx, 1))(d1)
        d2 = Reshape((embedding_dim, dimy, 1))(d2)
        
        if attention in [1,3]:    
            conv1 = merge([x,d1],mode='concat',concat_axis=1)
            conv2 = merge([y,d2],mode='concat',concat_axis=1)
        else:
            conv1,conv2 = x,y
        
    channel_1, channel_2  = [], []
        
    for dep in range(depth):
        
        filter_width = filter_widths[dep]
        
        if attention in [1,3]:
            conv1 = ZeroPadding2D((filter_width - 1, 0))(conv1)
            conv2 = ZeroPadding2D((filter_width - 1, 0))(conv2)
        
            if shared:
                conv = Conv2D(nb_filter=nb_filter, kernel_size = filter_length, activation='tanh',
                         data_format = 'channels_last',border_mode="valid")
                ques = conv(conv1)
                ans = conv(conv2)
                
            else:
                ques = Conv2D(nb_filter=nb_filter, kernel_size = filter_length, activation='relu',
                         data_format = 'channels_last',padding='same')(conv1)
                ans = Conv2D(nb_filter, kernel_size = filter_length, activation='relu',
                         data_format="channels_last",padding='same')(conv2)
                    
            
            if attention in [3]:
                    ques = Reshape((ques._keras_shape[1], ques._keras_shape[2]*ques._keras_shape[3]))(ques)
                    ans = Reshape((ans._keras_shape[1], ans._keras_shape[2]*ans._keras_shape[3]))(ans)
                    
                    rep_vec = ques._keras_shape[2]
                    
            #if attention in [3]:
                    
                    ans_T = Permute((2,1))(ans)
                    ques_T = Permute((2,1))(ques)
                        
                    attn2_mat = MatchScore(ques,ans)
                    
                    a1_row = Lambda(lambda a: K.sum(a,axis=1),output_shape=(attn2_mat._keras_shape[2],1))(attn2_mat)
                    a2_col = Lambda(lambda a: K.sum(a,axis=2),output_shape=(attn2_mat._keras_shape[1],1))(attn2_mat)
                    
                    a1_row = RepeatVector(rep_vec)(a1_row)
                    a2_col = RepeatVector(rep_vec)(a2_col)    
                    
                    attn_pool_1 = Merge(mode='mul')([a1_row, ques_T])
                    attn_pool_2 = Merge(mode='mul')([a2_col, ans_T])
                    #attn_pool_2 = Permute((2,1))(attn_pool_2)
                        
                    #h1 = Lambda(lambda a: K.sum(a,axis=1))(attn_pool_1)
                    #h2 = Lambda(lambda a: K.sum(a,axis=2))(attn_pool_2)
                    
                    conv1 = GlobalAveragePooling1D()(attn_pool_1)
                    conv2 = GlobalAveragePooling1D()(attn_pool_2)
                
            else:
                conv1 = GlobalMaxPooling2D()(ques)
                conv2 = GlobalMaxPooling2D()(ans)
                #conv1 = Flatten()MaxPooling2D()(ques)
                #conv2 = Flatten()MaxPooling2D()(ans)
        channel_1.append(conv1)
        channel_2.append(conv2)
    
    h1 = channel_1.pop(-1)
    if channel_1:
        h1 = merge([h1] + channel_1, mode="concat")

    h2 = channel_2.pop(-1)
    if channel_2:
        h2 = merge([h2] + channel_2, mode="concat")
    
    h =  Merge(mode="concat",name='h')([h1, h2])

    opt = keras.optimizers.adam(lr=opt_params[0],clipnorm=1.)

    score = Dense(2,activation='softmax',name='score')(h)
    model = Model([inpx, inpy],[score])
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    
    return model


def bcnn(embedding_matrix, dimx=50, dimy=50, nb_filter = 120, embedding_dim = 50, 
                      filter_length = (50,4), depth = 1, shared = 0,
                      opt_params = [0.0008,'adam']):

#if True:
    print 'Model Uses BCNN......'
    
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
        
        #filter_width = filter_widths[dep]
        
        #conv1 = ZeroPadding2D((filter_width - 1, 0))(conv1)
        #conv2 = ZeroPadding2D((filter_width - 1, 0))(conv2)
        
        if shared:
            conv = Conv2D(nb_filter=nb_filter, kernel_size = filter_length, activation='relu',
                        data_format = 'channels_last',border_mode="valid")
            ques = conv(conv1)
            ans = conv(conv2)
                
        else:
            ques = Conv2D(nb_filter=nb_filter, kernel_size = filter_length, activation='relu',
                    data_format = 'channels_last',border_mode="valid")(conv1)
            ans = Conv2D(nb_filter, kernel_size = filter_length, activation='relu',
                    data_format="channels_last",border_mode="valid")(conv2)
                    
            
        ques = Dropout(0.5)(ques)
        ans = Dropout(0.5)(ans)
        channel_1.append(GlobalMaxPooling2D()(ques))
        channel_2.append(GlobalMaxPooling2D()(ans))
        
        #channel_1.append(Reshape((ques._keras_shape[2]*ans._keras_shape[3]))(AveragePooling2D(4))(ques))
        #channel_2.appendFlatten()((AveragePooling2D(4))(ans))

       
    #reg1 = reg2 = 0.00002
    
    h1 = channel_1.pop(-1)
    if channel_1:
        h1 = merge([h1] + channel_1, mode="concat")

    h2 = channel_2.pop(-1)
    if channel_2:
        h2 = merge([h2] + channel_2, mode="concat")
    
    #h1 = Dropout(0.5)(h1)
    #h2 = Dropout(0.5)(h2)
    
    #reg2 = 0.00005
    
    h =  Merge(mode="concat",name='h')([h1, h2])
    #h = Dropout(0.2)(h)
    #h = Dense(50, kernel_regularizer=regularizers.l2(reg2),activation='relu')(h)
    #wrap = Dropout(0.5)(h)
    #wrap = Dense(64, activation='tanh')(h)   
    
    opt = keras.optimizers.adam(lr=opt_params[0],clipnorm=1.)
    
    score = Dense(2,activation='softmax',name='score')(h)
    model = Model([inpx, inpy],[score])
    model.compile( loss='categorical_crossentropy',optimizer=opt)
        #label = to_categorical(label)
    
    #model.fit([data_l,data_r],label,nb_epoch=nb_epoch,batch_size=batch_size,verbose=2)
    return model


def compute_euclidean_match_score(l_r):
    l, r = l_r
    denominator = 1. + K.sqrt(
        -2 * K.batch_dot(l, r, axes=[2, 2]) +
        K.expand_dims(K.sum(K.square(l), axis=2), 2) +
        K.expand_dims(K.sum(K.square(r), axis=2), 1)
    )
    denominator = K.maximum(denominator, K.epsilon())
    return 1. / denominator

def MatchScore(l, r, mode="euclidean"):
    if mode == "euclidean":
        return merge(
            [l, r],
            mode=compute_euclidean_match_score,
            output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1])
        )
