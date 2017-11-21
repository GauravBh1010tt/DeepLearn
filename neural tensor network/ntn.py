from attention import *
from keras import activations

class ntn_layer(Layer):
    def __init__(self,inp_size,out_size,activation='tanh', **kwargs):
        super(ntn_layer, self).__init__(**kwargs)
        self.k = out_size
        self.d = inp_size
        self.activation = activations.get(activation)
        self.test_out = 0
        
    def build(self,input_shape):
        
        self.W = self.add_weight(name='w',shape=(self.d, self.d, self.k),
                                      initializer='glorot_uniform',
                                  trainable=True)
        
        self.V = self.add_weight(name='v',shape=(self.k, self.d*2),
                                      initializer='glorot_uniform',
                                  trainable=True)
                                  
        self.b = self.add_weight(name='b',shape=(self.k,),
                                      initializer='zeros',
                                  trainable=True)
        
        self.U = self.add_weight(name='u',shape=(self.k,),
                                      #initializer='ones',
                                  #trainable=False
                initializer='glorot_uniform',trainable=True)
                                  
        super(ntn_layer, self).build(input_shape)     

    def call(self ,x ,mask=None):
        e1=x[0]
        e2=x[1]
      
        batch_size = K.shape(e1)[0]
        V_out, h, mid_pro = [],[],[]
        for i in range(self.k):
            V_out = K.dot(self.V[i],K.concatenate([e1,e2]).T)          
            temp = K.dot(e1,self.W[:,:,i])
            h = K.sum(temp*e2,axis=1)
            mid_pro.append(V_out+h+self.b[i])
     
        tensor_bi_product = K.concatenate(mid_pro,axis=0)

        tensor_bi_product = self.U*self.activation(K.reshape(
                            tensor_bi_product,(self.k,batch_size))).T
     
        self.test_out = K.shape(tensor_bi_product)
        
        return tensor_bi_product


    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],self.k)
        