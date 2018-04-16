# Attention Based CNN (ABCNN)
### This is the implementation of [ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](https://arxiv.org/pdf/1512.05193.pdf). 
The deep model used is CNN with attentive convolution and reweighted pooling:
<img src="https://github.com/GauravBh1010tt/DeepLearn/blob/master/Attention_Based_CNN%20(ABCNN)/abcnn.JPG" width="738">

# Dependencies
#### The required dependencies are mentioned in requirement.txt. I will also use **[dl-text](https://github.com/GauravBh1010tt/DL-text)** modules for preparing the datasets. If you haven't use it, please do have a quick look at it. 

```python
$ pip install -r requirements.txt
```

# Usage
The script to run the codes are given in ```main.py```. For a quick run, download all contents in a single folder and run:
```python
$ python main.py
```
You can also use the Python ```Idle``` to run the modules as follows:
```python
>>> import sys
>>> sys.path.append("..\_deeplearn_utils")

>>> import model_abcnn as model
>>> from dl_text.metrics import eval_metric
>>> from dl_text import dl
>>> import wiki_utils as wk

>> wordVec = 'path_to_Word2Vec(300 dim)/GoogleNews-vectors-negative300.bin.gz'

############################ DEFINING MODEL ############################

>>> lrmodel = model.bcnn # bcnn for basic cnn, abcnn for attentive cnn
>>> model_name = lrmodel.func_name

################### DEFINING HYPERPARAMETERS ###################

>>> dimx = 50 # number of words in sentence 1
>>> dimy = 50 # number of words in sentence 2 
>>> embedding_dim = 50
>>> nb_filter = 120
>>> filter_length = (50,4)
>>> depth = 1
>>> nb_epoch = 3
>>> shared = 0
>>> opt_params = [0.001,'adam']
```
For evaluating the performance of the model I will use WikiQA (answer sentence selection). This dataset can be further processed using **[dl-text](https://github.com/GauravBh1010tt/DL-text)**. Prepare the datasets as:

```python
>>> ques, ans, label_train, train_len, test_len,\
         wordVec_model, res_fname, pred_fname, feat_train, feat_test = wk.load_wiki(model_name, glove_fname)
>>> data_l , data_r, embedding_matrix = dl.process_data(ques, ans,
                                                 wordVec_model,dimx=dimx,
                                                 dimy=dimy,vocab_size=vocab_size,
                                                 embedding_dim=embedding_dim)

>>> X_train_l,X_test_l,X_dev_l,X_train_r,X_test_r,X_dev_r = wk.prepare_train_test(data_l,data_r,
                                                                           train_len,test_len)

```
For basic CNN (**BCNN**) model, train the architecture as:
```python
>>> lrmodel = lrmodel(embedding_matrix, dimx=dimx, dimy=dimy, nb_filter = nb_filter, embedding_dim = embedding_dim, 
                      filter_length = filter_length, depth = depth, shared = shared,
                      opt_params = opt_params)
    
>>> print '\n', model_name,'model built \n'
>>> lrmodel.fit([X_train_l, X_train_r],label_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=2)
>>> map_val, mrr_val = eval_metric(lrmodel, X_test_l, X_test_r, res_fname, pred_fname)
>>> print 'MAP : ',map_val,' MRR : ',mrr_val
MAP : 64.23 MRR : 66.74
```

For attentive CNN (**ABCNN**) model, train the architecture as:
```python
>>> lrmodel = lrmodel(embedding_matrix, dimx=dimx, dimy=dimy, nb_filter = nb_filter, embedding_dim = embedding_dim, 
                      filter_length = filter_length, depth = depth, shared = shared,
                      opt_params = opt_params)
    
 >>> print '\n',model_name,'model built \n'
 >>> lrmodel.fit([X_train_l, X_train_r],label_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=2)
 >>> map_val, mrr_val = eval_metric(lrmodel, X_test_l, X_test_r, res_fname, pred_fname)
 >>> print 'MAP : ',map_val,' MRR : ',mrr_val
 MAP : 66.78 MRR : 67.34
 ```
