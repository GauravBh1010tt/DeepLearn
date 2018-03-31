# Manhattan Siamese LSTM
### This is the implementation of [Siamese Recurrent Architectures for Learning Sentence Similarity](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf). 
The deep model used is LSTM with manhattan scoring parameters:
<img src="https://github.com/GauravBh1010tt/DeepLearn/blob/master/MaLSTM%20(Siamese)/malstm.JPG" width="438">

# Dependencies
#### The required dependencies are mentioned in requirement.txt. I will also use **[dl-text](https://github.com/GauravBh1010tt/DL-text)** modules for preparing the datasets. If you haven't use it, please do have a quick look at it. 

```python
$ pip install -r requirements.txt
```

# Usage
The script to run the codes are given in ```main.py```. You can also use the Python ```Idle``` to run the modules.

```python
>>> import model_Siam_LSTM as model
>>> from dl_text.metrics import eval_sick
>>> from dl_text import dl
>>> import sick_utils as sick

>>> glove_fname = 'path to word_vector file/glove.6B.50d.txt'

############################ DEFINING MODEL ############################

>>> lrmodel = model.S_LSTM
>>> model_name = lrmodel.func_name

################### DEFINING HYPERPARAMETERS ###################

>>> dimx = 50 # number of words in sentence 1
>>> dimy = 50 # number of words in sentence 2 
>>> batch_size = 50
>>> vocab_size = 8000
>>> embedding_dim = 50
>>> nb_filter = 120
>>> filter_length = (50,4)
>>> depth = 1
>>> nb_epoch = 3
>>> num_tensor_slices = 4
```
For evaluating the performance of the model I will use TrecQA dataset. The reason I am using this dataset is the dataset mentioned in the paper is not publically available. This dataset can be further processed using **[dl-text](https://github.com/GauravBh1010tt/DL-text)**. Prepare the datasets as:

```python
>>> ques, ans, label_train, train_len, test_len, wordVec_model, \
        res_fname, pred_fname, feat_train, feat_test = trec.load_trec(model_name, glove_fname)
            
>>> data_l , data_r, embedding_matrix = dl.process_data(ques, ans,
                                                 wordVec_model,dimx=dimx,
                                                 dimy=dimy,vocab_size=vocab_size,
                                                 embedding_dim=embedding_dim)

>>> X_train_l,X_test_l,X_dev_l,X_train_r,X_test_r,X_dev_r = trec.prepare_train_test(data_l,data_r,
                                                                           train_len,test_len)
```

The **CNN model with tensor parameters** can be trained as:
```python
>>> lrmodel = lrmodel(embedding_matrix, dimx=dimx, dimy=dimy, nb_filter = nb_filter, 
                      num_slices = num_tensor_slices, embedding_dim = embedding_dim, 
                      filter_length = filter_length, vocab_size = vocab_size, depth = depth)
    
>>> lrmodel.fit([X_train_l, X_train_r],
                label_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=2)
>>> map_val, mrr_val = eval_metric(lrmodel, X_test_l, X_test_r, res_fname, pred_fname)
>>> print 'MAP : ',map_val,' MRR : ',mrr_val
MAP :  0.643286925881  MRR :  70.4599673203
```
