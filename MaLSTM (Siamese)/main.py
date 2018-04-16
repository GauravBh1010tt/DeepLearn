"""
** deeplean-ai.com **
** dl-lab **
created by :: GauravBh1010tt
"""

import sys
sys.path.append("..\_deeplearn_utils")

import model_Siam_LSTM as model
from dl_text.metrics import eval_sick
from dl_text import dl
import sick_utils as sick
import numpy as np

lrmodel = model.S_LSTM
model_name = lrmodel.func_name

embedding_dim = 300
LSTM_neurons = 50
dimx = 30
dimy = 30
vocab_size = 8000
batch_size = 32
epochs = 3

wordVec = 'path_to_Word2Vec(300 dim)/GoogleNews-vectors-negative300.bin.gz'
wordVec = None
sent1, sent2, train_len, test_len, train_score, test_score, wordVec_model, pred_fname = sick.load_sick(model_name, wordVec)

data_l, data_r, embedding_matrix = dl.process_data(sent1, sent2,
                                                 wordVec_model,dimx=dimx,
                                                 dimy=dimy,vocab_size=vocab_size,
                                                 embedding_dim=embedding_dim)  

X_train_l,X_test_l,X_dev_l,X_train_r,X_test_r,X_dev_r = dl.prepare_train_test(data_l,data_r,
                                                                              train_len,test_len)        
   
print '\n', model_name,'model built \n'
    
lrmodel = lrmodel(dimx = dimx, dimy = dimy, embedding_matrix=embedding_matrix, 
                   LSTM_neurons = LSTM_neurons)
lrmodel.fit([X_train_l,X_train_r],
                train_score,                 
                 nb_epoch=epochs,
                 batch_size=batch_size,verbose=1)

print '\n evaluating performance \n'

sp_coef, per_coef, mse = eval_sick(lrmodel, X_test_l, X_test_r, test_score)
print 'spearman coef :',sp_coef
print 'pearson coef :',per_coef
print 'mse :',mse
