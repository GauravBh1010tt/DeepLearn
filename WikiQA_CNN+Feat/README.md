# WikiQA - CNN+Feat
### This is the implementation of [WIKIQA: A Challenge Dataset for Open-Domain Question Answering](https://aclweb.org/anthology/D15-1237). The deep model used is CNN with external features:
![models](https://github.com/GauravBh1010tt/DeepLearn/blob/master/WikiQA_CNN%2BFeat/im1.JPG)

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

>>> from dl_text import dl
>>> import model
>>> import wiki_utils as wk
>>> from dl_text.metrics import eval_metric

>>> glove_fname = 'path to word_vector file/glove.6B.50d.txt'

############################ DEFINING MODEL ############################

>>> lrmodel = model.cnn   # for CNN - model.cnn , for CNN_FT - model.cnn_ft
>>> model_name = lrmodel.func_name

################### DEFINING HYPERPARAMETERS ###################

>>> dimx = 60 # number of words in question
>>> dimy = 60 # number of words in answer
>>> dimft = 44 # number of external features used
>>> batch_size = 50
>>> vocab_size = 8000
>>> embedding_dim = 50
>>> nb_filter = 120
>>> filter_length = (50,4)
>>> depth = 1
>>> nb_epoch = 3
```
I have extracted the external features and stores in the ```Extracted_Features``` folder. You can compute these features using **[dl-text](https://github.com/GauravBh1010tt/DL-text)**. Prepare the datasets as:

```python
>>> ques, ans, label_train, train_len, test_len, wordVec_model, \
        res_fname, pred_fname, feat_train, feat_test = wk.load_wiki(model_name, glove_fname)
            
>>> data_l , data_r, embedding_matrix = dl.process_data(ques, ans,
                                                 wordVec_model,dimx=dimx,
                                                 dimy=dimy,vocab_size=vocab_size,
                                                 embedding_dim=embedding_dim)

>>> X_train_l,X_test_l,X_dev_l,X_train_r,X_test_r,X_dev_r = wk.prepare_train_test(data_l,data_r,
                                                                           train_len,test_len)
```

The **basic CNN model** can be trained as:
```python
>>> lrmodel = lrmodel(embedding_matrix, dimx=dimx, dimy=dimy, nb_filter = nb_filter, 
                      embedding_dim = embedding_dim, filter_length = filter_length,
                      vocab_size = vocab_size, depth = depth)
    
>>> lrmodel.fit([X_train_l, X_train_r],
                label_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=2)
>>> map_val, mrr_val = eval_metric(lrmodel, X_test_l, X_test_r, res_fname, pred_fname)
>>> print 'MAP : ',map_val,' MRR : ',mrr_val
MAP :  0.614643347051  MRR :  62.7487099092
```
The results are comparable to the results mentioned in the paper. For **CNN model with external features**, use 


```python
>>> lrmodel = lrmodel(embedding_matrix, dimx=dimx, dimy=dimy, dimft=dimft, nb_filter = nb_filter, 
                      embedding_dim = embedding_dim, filter_length = filter_length,
                      vocab_size = vocab_size, depth = depth)
>>> lrmodel.fit([X_train_l,X_train_r,feat_train],
                    label_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=2)
>>> map_val, mrr_val = eval_metric(lrmodel, X_test_l,
                    X_test_r, res_fname, pred_fname, feat_test=feat_test)
    
>>> print 'MAP : ',map_val,' MRR : ',mrr_val
MAP :  0.65945637298  MRR :  66.184657532
```
