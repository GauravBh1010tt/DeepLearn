# -*- coding: utf-8 -*-

"""
Created on Tue Mar 07 11:48:18 2017

@author: Gaurav Bhatt
Email - gauravbhatt.cs.iitr@gmail.com
"""

import sys
import math
import random
import warnings
import numpy as np
from sklearn import svm
import keras.backend as K
from keras.models import Model
from theano import tensor as T
import matplotlib.pyplot as plt
from keras.layers import Input, Merge
from keras.engine.topology import Layer
from sklearn.metrics import accuracy_score
from keras.layers.core import  Activation, Dense

warnings.simplefilter("ignore")

hdim = 50
h_loss = 50
hdim_deep = 500
hdim_deep2 = 300
nb_epoch = 40
batch_size = 100
dimx = 392
dimy = 392
lamda = 0.02
loss_type = 2 # 1 - l1+l2+l3-L4; 2 - l2+l3-L4; 3 - l1+l2+l3 , 4 - l2+l3

def svm_classifier(train_x, train_y, valid_x, valid_y, test_x, test_y):
    
    clf = svm.LinearSVC()
    #print train_x.shape,train_y.shape
    clf.fit(train_x,train_y)
    pred = clf.predict(valid_x)
    va = accuracy_score(np.ravel(valid_y),np.ravel(pred))
    pred = clf.predict(test_x)
    ta = accuracy_score(np.ravel(test_y),np.ravel(pred))
    return va, ta

def split(train_l,train_r,label,ratio):
    
    total = train_l.shape[0]
    train_samples = int(total*(1-ratio))
    test_samples = total-train_samples
    tr_l,tst_l,tr_r,tst_r,l_tr,l_tst=[],[],[],[],[],[]
    dat=random.sample(range(total),train_samples)
    for a in dat:
        tr_l.append(train_l[a,:])
        tr_r.append(train_r[a,:])
        l_tr.append(label[a])
        
    for i in range(test_samples):
        if i not in dat:
            tst_l.append(train_l[i,:])
            tst_r.append(train_r[i,:])
            l_tst.append(label[i])
            
    tr_l = np.array(tr_l)
    tr_r = np.array(tr_r)
    tst_l = np.array(tst_l)
    tst_r = np.array(tst_r)
    l_tr = np.array(l_tr)
    l_tst = np.array(l_tst)
    
    return tr_l,tst_l,tr_r,tst_r,l_tr,l_tst
    
    
class ZeroPadding(Layer):
    def __init__(self, **kwargs):
        super(ZeroPadding, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.zeros_like(x)

    def get_output_shape_for(self, input_shape):
        return input_shape

class MultiplyBy2(Layer):
    def __init__(self, **kwargs):
        super(MultiplyBy2, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return 2*x

    def get_output_shape_for(self, input_shape):
        return input_shape


class CorrnetCost(Layer):
    def __init__(self,lamda, **kwargs):
        super(CorrnetCost, self).__init__(**kwargs)
        self.lamda = lamda

    def cor(self,y1, y2, lamda):
        y1_mean = K.mean(y1, axis=0)
        y1_centered = y1 - y1_mean
        y2_mean = K.mean(y2, axis=0)
        y2_centered = y2 - y2_mean
        corr_nr = K.sum(y1_centered * y2_centered, axis=0)
        corr_dr1 = K.sqrt(T.sum(y1_centered * y1_centered, axis=0) + 1e-8)
        corr_dr2 = K.sqrt(T.sum(y2_centered * y2_centered, axis=0) + 1e-8)
        corr_dr = corr_dr1 * corr_dr2
        corr = corr_nr / corr_dr
        return K.sum(corr) * lamda

    def call(self ,x ,mask=None):
        h1=x[0]
        h2=x[1]

        corr = self.cor(h1,h2,self.lamda)

        #self.add_loss(corr,x)
        #we output junk but be sure to use it for the loss to be added
        return corr

    def get_output_shape_for(self, input_shape):
        #print input_shape[0][0]
        return (input_shape[0][0],input_shape[0][1])

def corr_loss(y_true, y_pred):
    #print y_true.type,y_pred.type
    #return K.zeros_like(y_pred)
    return y_pred

def project(model,inp):
    m = model.predict([inp[0],inp[1]])
    return m[2]

def reconstruct_from_left(model,inp):
    img_inp = inp.reshape((28,14))
    f, axarr = plt.subplots(1,2,sharey=False)
    pred = model.predict([inp,np.zeros_like(inp)])
    img = pred[0].reshape((28,14))
    axarr[0].imshow(img_inp)
    axarr[1].imshow(img)

def reconstruct_from_right(model,inp):
    img_inp = inp.reshape((28,14))
    f, axarr = plt.subplots(1,2,sharey=False)
    pred = model.predict([np.zeros_like(inp),inp])
    img = pred[1].reshape((28,14))
    axarr[1].imshow(img_inp)
    axarr[0].imshow(img)
    
def sum_corr(model):
    view1 = np.load("test_v1.npy")
    view2 = np.load("test_v2.npy")
    x = project(model,[view1,np.zeros_like(view1)])
    y = project(model,[np.zeros_like(view2),view2])
    print "test correlation"
    corr = 0
    for i in range(0,len(x[0])):
		x1 = x[:,i] - (np.ones(len(x))*(sum(x[:,i])/len(x)))
		x2 = y[:,i] - (np.ones(len(y))*(sum(y[:,i])/len(y)))
		nr = sum(x1 * x2)/(math.sqrt(sum(x1*x1))*math.sqrt(sum(x2*x2)))
		corr+=nr
    print corr
 
def transfer(model):
    view1 = np.load("test_v1.npy")
    view2 = np.load("test_v2.npy")
    labels = np.load("test_l.npy")
    view1 = project(model,[view1,np.zeros_like(view1)])
    view2 = project(model,[np.zeros_like(view2),view2])
    
    perp = len(view1)/5
    print "view1 to view2"
    acc = 0
    for i in range(0,5):
		test_x = view2[i*perp:(i+1)*perp]
		test_y = labels[i*perp:(i+1)*perp]
		if i==0:
			train_x = view1[perp:len(view1)]
			train_y = labels[perp:len(view1)]
		elif i==4:
			train_x = view1[0:4*perp]
			train_y = labels[0:4*perp]
		else:
			train_x1 = view1[0:i*perp]
			train_y1 = labels[0:i*perp]
			train_x2 = view1[(i+1)*perp:len(view1)]
			train_y2 = labels[(i+1)*perp:len(view1)]
			train_x = np.concatenate((train_x1,train_x2))
			train_y = np.concatenate((train_y1,train_y2))
       
		va, ta = svm_classifier(train_x, train_y, test_x, test_y, test_x, test_y)
		acc += ta
    print acc/5
    print "view2 to view1"

    acc = 0
    for i in range(0,5):
		test_x = view1[i*perp:(i+1)*perp]
		test_y = labels[i*perp:(i+1)*perp]
		if i==0:
			train_x = view2[perp:len(view1)]
			train_y = labels[perp:len(view1)]
		elif i==4:
			train_x = view2[0:4*perp]
			train_y = labels[0:4*perp]
		else:
			train_x1 = view2[0:i*perp]
			train_y1 = labels[0:i*perp]
			train_x2 = view2[(i+1)*perp:len(view1)]
			train_y2 = labels[(i+1)*perp:len(view1)]
			train_x = np.concatenate((train_x1,train_x2))
			train_y = np.concatenate((train_y1,train_y2))
		va, ta = svm_classifier(train_x, train_y, test_x, test_y, test_x, test_y)
		acc += ta
    print acc/5

def prepare_data():
    data_l = np.load('data_l.npy')
    data_r = np.load('data_r.npy')
    label = np.load('data_label.npy')
    X_train_l, X_test_l, X_train_r, X_test_r,y_train,y_test = split(data_l,data_r,label,ratio=0.0)
    return X_train_l, X_train_r

def buildModel(loss_type,lamda):

    inpx = Input(shape=(dimx,))
    inpy = Input(shape=(dimy,))

    hx = Dense(hdim_deep,activation='sigmoid')(inpx)
    hx = Dense(hdim_deep2, activation='sigmoid',name='hid_l1')(hx)
    hx = Dense(hdim, activation='sigmoid',name='hid_l')(hx)
    
    hy = Dense(hdim_deep,activation='sigmoid')(inpy)
    hy = Dense(hdim_deep2, activation='sigmoid',name='hid_r1')(hy)
    hy = Dense(hdim, activation='sigmoid',name='hid_r')(hy)

    #h = Activation("sigmoid")( Merge(mode="sum")([hx,hy]) )
    h =  Merge(mode="sum")([hx,hy]) 
    
    #recx = Dense(hdim_deep,activation='sigmoid')(h)
    recx = Dense(dimx)(h)
    #recy = Dense(hdim_deep,activation='sigmoid')(h)
    recy = Dense(dimy)(h)
    
    branchModel = Model( [inpx,inpy],[recx,recy,h])

    #inpx = Input(shape=(dimx,))
    #inpy = Input(shape=(dimy,))

    [recx1,recy1,h1] = branchModel( [inpx, ZeroPadding()(inpy)])
    [recx2,recy2,h2] = branchModel( [ZeroPadding()(inpx), inpy ])

    #you may probably add a reconstruction from combined
    [recx3,recy3,h] = branchModel([inpx, inpy])

    corr=CorrnetCost(-lamda)([h1,h2])
    
    if loss_type == 1:
        model = Model( [inpx,inpy],[recy1,recx2,recx3,recx1,recy2,recy3,corr])
        model.compile( loss=["mse","mse","mse","mse","mse","mse",corr_loss],optimizer="rmsprop")
    elif loss_type == 2:
        model = Model( [inpx,inpy],[recy1,recx2,recx1,recy2,corr])
        model.compile( loss=["mse","mse","mse","mse",corr_loss],optimizer="rmsprop")
    elif loss_type == 3:
        model = Model( [inpx,inpy],[recy1,recx2,recx3,recx1,recy2,recy3])
        model.compile( loss=["mse","mse","mse","mse","mse","mse"],optimizer="rmsprop")
    elif loss_type == 4:
        model = Model( [inpx,inpy],[recy1,recx2,recx1,recy2])
        model.compile( loss=["mse","mse","mse","mse"],optimizer="rmsprop")

    return model, branchModel

def trainModel(model,data_left,data_right,loss_type,nb_epoch,batch_size):

    X_train_l = data_left
    X_train_r = data_right
    #y_train = np_utils.to_categorical(y_train, nb_classes)
    #y_test = np_utils.to_categorical(y_test, nb_classes)
    
    data_l = np.load('data_l.npy')
    data_r = np.load('data_r.npy')
    label = np.load('data_label.npy')
    X_train_l, X_test_l, X_train_r, X_test_r,y_train,y_test = split(data_l,data_r,label,ratio=0.01)
    
    print 'data split'
    if loss_type == 1:
        print 'L_Type: l1+l2+l3-L4   h_dim:',hdim,'  lamda:',lamda
        model.fit([X_train_l,X_train_r], [X_train_r,X_train_l,X_train_l,X_train_l,X_train_r,X_train_r,np.zeros((X_train_l.shape[0],h_loss))],
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,verbose=0)
    elif loss_type == 2:
        print 'L_Type: l2+l3-L4   h_dim:',hdim,'   hdim_deep',hdim_deep,'  lamda:',lamda
        model.fit([X_train_l,X_train_r], [X_train_r,X_train_l,X_train_l,X_train_r,np.zeros((X_train_l.shape[0],h_loss))],
              nb_epoch=nb_epoch,
              batch_size=batch_size,verbose=0)
    elif loss_type == 3:
        print 'L_Type: l1+l2+l3   h_dim:',hdim,'  lamda:',lamda
        model.fit([X_train_l,X_train_r], [X_train_r,X_train_l,X_train_l,X_train_l,X_train_r,X_train_r],
              nb_epoch=nb_epoch,
              batch_size=batch_size,verbose=0)
    elif loss_type == 4:
        print 'L_Type: l2+l3   h_dim:',hdim,'  lamda:',lamda
        model.fit([X_train_l,X_train_r], [X_train_r,X_train_l,X_train_l,X_train_r],
              nb_epoch=nb_epoch,
              batch_size=batch_size,verbose=0)
    #score = m.evaluate([X_test_l,X_test_r], [X_test_l,X_test_l,X_test_r,X_test_r,np.zeros((X_test_l.shape[0],hdim))],
    #                  batch_size=100)
    #print score

def testModel(b_model):
    transfer(b_model)
    sum_corr(b_model)

left_view, right_view = prepare_data()
model,branchModel = buildModel(loss_type=loss_type,lamda=lamda)
trainModel(model=model, data_left=left_view, data_right = right_view, 
           loss_type=loss_type,nb_epoch=nb_epoch,batch_size=batch_size)
testModel(branchModel)
