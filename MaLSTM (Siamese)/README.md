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
The script to run the codes are given in ```main.py```. For a quick run, download all contents in a single folder and run:
```python
$ python main.py
```
You can also use the Python ```Idle``` to run the modules as follows:
```python
>>> import sys
>>> sys.path.append("..\_deeplearn_utils")

>>> import model_Siam_LSTM as model
>>> from dl_text.metrics import eval_sick
>>> from dl_text import dl
>>> import sick_utils as sick

>> wordVec = 'path_to_Word2Vec(300 dim)/GoogleNews-vectors-negative300.bin.gz'

############################ DEFINING MODEL ############################

>>> lrmodel = model.S_LSTM
>>> model_name = lrmodel.func_name

################### DEFINING HYPERPARAMETERS ###################

>>> dimx = 30 # number of words in sentence 1
>>> dimy = 30 # number of words in sentence 2 
>>> embedding_dim = 300
>>> LSTM_neurons = 50
>>> vocab_size = 8000
>>> batch_size = 32
>>> epochs = 3
```
For evaluating the performance of the model I will use SICK dataset (sentence textual similarity). This dataset can be further processed using **[dl-text](https://github.com/GauravBh1010tt/DL-text)**. Prepare the datasets as:

```python
>>> sent1, sent2, train_len, test_len, train_score, test_score, wordVec_model,\
                                                                 pred_fname = sick.load_sick(model_name, wordVec)
            
>>> data_l , data_r, embedding_matrix = dl.process_data(sent1, sent2,
                                                 wordVec_model,dimx=dimx,
                                                 dimy=dimy,vocab_size=vocab_size,
                                                 embedding_dim=embedding_dim)

>>> X_train_l,X_test_l,X_dev_l,X_train_r,X_test_r,X_dev_r = trec.prepare_train_test(data_l,data_r,
                                                                           train_len,test_len)
```

The **MaLSTM model with mahattan scoring parameters** can be trained as:
```python
>>> lrmodel = lrmodel(dimx = dimx, dimy = dimy, embedding_matrix=embedding_matrix, 
                      LSTM_neurons = LSTM_neurons)
>>> lrmodel.fit([X_train_l,X_train_r], train_score, nb_epoch=epochs, batch_size=batch_size, verbose=2)

>>> sp_coef, per_coef, mse = eval_sick(lrmodel, X_test_l, X_test_r, test_score)
>>> print 'spearman coef :',sp_coef, ' \n pearson coef :',per_coef, ' \n mse :',mse
spearman coef : 0.747002297032
pearson coef : 0.817191946996
mse : 0.375428835647
```
