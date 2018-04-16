"""
** deeplean-ai.com **
** dl-lab **
created by :: GauravBh1010tt
"""

import sys
sys.path.append("..\_deeplearn_utils")

import model_sim as model
import trec_utils as trec
from dl_text.metrics import eval_metric
from dl_text import dl

glove_fname = 'K:/workspace/neural network/Trec_QA-master/glove.6B.50d.txt'

################### DEFINING MODEL AND PREDICTION FILE ###################

lrmodel = model.cnn_sim_ft
model_name = lrmodel.func_name

################### DEFINING HYPERPARAMETERS ###################

dimx = 50
dimy = 50
dimft = 44
batch_size = 70
vocab_size = 10000
embedding_dim = 50
nb_filter = 120
filter_length = (50,4)
depth = 1
nb_epoch = 4
    
ques, ans, label_train, train_len, test_len,\
         wordVec_model, res_fname, pred_fname, feat_train, feat_test = trec.load_trec(model_name, glove_fname)
data_l , data_r, embedding_matrix = dl.process_data(ques, ans,
                                                 wordVec_model,dimx=dimx,
                                                 dimy=dimy,vocab_size=vocab_size,
                                                 embedding_dim=embedding_dim)

X_train_l,X_test_l,X_dev_l,X_train_r,X_test_r,X_dev_r = trec.prepare_train_test(data_l,data_r,
                                                                           train_len,test_len)

if model_name == 'cnn_sim_ft':
    lrmodel = lrmodel(embedding_matrix, dimx=dimx, dimy=dimy, dimft=dimft, nb_filter = nb_filter,
                      embedding_dim = embedding_dim,filter_length = filter_length, vocab_size = vocab_size,
                      depth = depth)
    
    print '\n',model_name,'model built \n'
    lrmodel.fit([X_train_l, X_train_r,feat_train],label_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=2)
    map_val, mrr_val = eval_metric(lrmodel, X_test_l, X_test_r, res_fname, pred_fname, feat_test=feat_test)

else:
    lrmodel = lrmodel(embedding_matrix, dimx=dimx, dimy=dimy, nb_filter = nb_filter, embedding_dim = embedding_dim,
                      filter_length = filter_length, vocab_size = vocab_size, depth = depth)
    
    print '\n', model_name,'model built \n'
    lrmodel.fit([X_train_l, X_train_r],label_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=2)
    map_val, mrr_val = eval_metric(lrmodel, X_test_l, X_test_r, res_fname, pred_fname)

    
print 'MAP : ',map_val,' MRR : ',mrr_val
