"""
** deeplean-ai.com **
** dl-lab **
created by :: GauravBh1010tt
"""

import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints , activations

np.random.seed(21)

class ntn(Layer):
    def __init__(self,inp_size, out_size, activation='tanh', **kwargs):
        super(ntn, self).__init__(**kwargs)
        self.k = out_size
        self.d = inp_size
        self.activation = activations.get(activation)
        self.test_out = 0
        
    def build(self,input_shape):
        
        self.W = self.add_weight(name='w',shape=(self.k, self.d, self.d),
                                      initializer='glorot_uniform',trainable=True)
                                      #initializer='ones',trainable=False)
        
        self.V = self.add_weight(name='v',shape=(self.k, self.d*2),
                                      initializer='glorot_uniform',trainable=True)
                                    #initializer='ones',trainable=False)
                                  
        #self.b = self.add_weight(name='b',shape=(self.k,1),
        #                              initializer='glorot_uniform',trainable=True)
#                                    initializer='ones',trainable=False)
        
        self.U = self.add_weight(name='u',shape=(self.k,1),
#                                      initializer='ones',trainable=False)
                initializer='glorot_uniform',trainable=True)
                                  
        super(ntn, self).build(input_shape)     

    def call(self , x, mask=None):
        
        e1=x[0].T
        e2=x[1].T
        
        batch_size = K.shape(x[0])[0]
        sim = []
        V_out = K.dot(self.V, K.concatenate([e1,e2],axis=0))     

        for i in range(self.k): 
            temp = K.batch_dot(K.dot(e1.T,self.W[i,:,:]),e2.T,axes=1)
            sim.append(temp)
        sim=K.reshape(sim,(self.k,batch_size))

        tensor_bi_product = self.activation(V_out+sim)
        tensor_bi_product = K.dot(self.U.T, tensor_bi_product).T

        return tensor_bi_product
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)      
    

class Similarity(Layer):
    
    def __init__(self, v_dim, kernel_regularizer=None, **kwargs):
        self.v_dim = v_dim
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        super(Similarity, self).__init__(**kwargs)

    def build(self,input_shape):
        self.W = self.add_weight(name='w',shape=(self.v_dim, self.v_dim),
                                      initializer='glorot_uniform',
                                      regularizer=self.kernel_regularizer,
                                  trainable=True)     
                                
        super(Similarity, self).build(input_shape)

    def call(self, data, mask=None):
        v1 = data[0]
        v2 = data[1]
        sim = K.dot(v1,self.W)
        sim = K.batch_dot(sim,v2,axes=1)
        return sim

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)