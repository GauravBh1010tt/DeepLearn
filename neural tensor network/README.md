
This is a implementation of Neural Tensor Network as described in [Reasoning With Neural Tensor Networks
for Knowledge Base Completion](http://cs.stanford.edu/~danqi/papers/nips2013.pdf). 
#### For detailed description please refer to my blog post [NEURAL TENSOR NETWORK: EXPLORING RELATIONS AMONG TEXT ENTITIES](http://deeplearn-ai.com/2017/11/21/neural-tensor-network-exploring-relations-among-text-entities/).

Neural Tensor Layer is defined as 

![img](https://user-images.githubusercontent.com/22491381/33068490-8cd268c6-ced7-11e7-9ca3-7e2dd7a29f86.JPG)

### Dependencies
This implementation uses Python 2.7, Keras (2.0 or above) with Theano backend, and Scikit Learn. 

### Data Used
Download the data files for wordbase and freebase dataset - [Data](https://drive.google.com/open?id=15JZNzH9J7cN1Qil87z7bhDYhmKa8TYvt). Extract the files and keep them in the same folder with the files in this repository (inside the data folder).

### Code Usage

```python
>>> import ntn_input
>>> from ntn import *
>>> from ntn_model import *
>>> negative_samples = 1
>>> e1,e2,labels_train,t1,t2,labels_dev,num_relations = prepare_data(negative_samples)
>>> e, t, labels_train, labels_dev = aggregate(e1,e2,labels_train,t1,t2,labels_dev,num_relations)
>>> model = build_model(num_relations, tensor_slices) # tensor_slices = number of tensor parameters
>>> model.fit(e,labels_train,
                 nb_epoch=10,
                 batch_size=100,verbose=1)
```
