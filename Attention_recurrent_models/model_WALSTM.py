"""
** deeplean-ai.com **
** dl-lab **
created by :: GauravBh1010tt
"""

import keras
from keras.models import Model
from keras import backend as K
from keras.layers.core import Dense, Reshape, Permute, Activation
from keras.layers import Input, merge, ZeroPadding2D, RepeatVector, LSTM, Bidirectional, GlobalAveragePooling2D,GlobalMaxPooling1D,GlobalAveragePooling1D,ZeroPadding1D,AveragePooling1D, GlobalMaxPooling2D, Dropout, Merge, Conv1D, Lambda, Flatten,  Conv2D, MaxPooling2D, MaxPooling1D, UpSampling2D, Convolution2D, TimeDistributed


from dl_text.dl import word2vec_embedding_layer

def WA_LSTM(embedding_matrix, dimx=50, dimy=50, nb_filter = 120, embedding_dim = 50, 
                      filter_length = (50,4), depth = 1, shared = 0,LSTM_neurons=64,word_level=1,
                      opt_params = [0.0008,'adam']):

    print 'Model Uses Attenion+LSTM......'
    
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    inpy = Input(shape=(dimy,),dtype='int32',name='inpy')
    
    x = word2vec_embedding_layer(embedding_matrix,train=False)(inpx)
    y = word2vec_embedding_layer(embedding_matrix,train=False)(inpy)
    
    #x = Permute((2,1))(x)
    #y = Permute((2,1))(y)

    channel_1,channel_2 = [],[]
    
    for dep in range(depth):
        
        #filter_width = filter_widths[dep]
        
        #conv1 = ZeroPadding2D((filter_width - 1, 0))(conv1)
        #conv2 = ZeroPadding2D((filter_width - 1, 0))(conv2)
        
        if shared:
            shared_lstm = Bidirectional(LSTM(LSTM_neurons,return_sequences=True),merge_mode='concat')   
            ques = shared_lstm(x)
            ans = shared_lstm(y)
                
        else:
            ques = Bidirectional(LSTM(LSTM_neurons,return_sequences=True),merge_mode='concat')(x)
            ans = Bidirectional(LSTM(LSTM_neurons,return_sequences=True),merge_mode='concat')(y)
            
##############     word - level attention       #########################
        
        if word_level:
            q_vec = TimeDistributed(Dense(1))(ques)
        else:
            q_vec = Dense(1)(ques)
            q_vec = RepeatVector(dimx)(q_vec)
            
            
        a_vec = TimeDistributed(Dense(1))(ans)
        m = Merge(mode='sum')([q_vec,a_vec])
        m = Activation(activation='tanh')(m)
        s = TimeDistributed(Dense(1,activation='softmax'))(m)
        ans_f = Merge(mode='mul')([ans,s])
        
        ques = Dropout(0.5)(ques)
        ans = Dropout(0.5)(ans)
        channel_1.append(GlobalMaxPooling1D()(ques))
        channel_2.append(GlobalMaxPooling1D()(ans_f))
        
        x = MaxPooling1D()(ques)
        y = MaxPooling1D()(ans)

       
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

    return model