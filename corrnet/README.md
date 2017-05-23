
# CorrNet

This is an implementation of Correlational Neural Network (CorrNet) described in the following paper : Sarath Chandar, Mitesh M Khapra, Hugo Larochelle, Balaraman Ravindran. [Correlational Neural Networks](https://arxiv.org/pdf/1504.07225.pdf). 

### For detailed description please refer to my blog post [COMMON REPRESENTATION LEARNING USING DEEP CORRNET](https://deeplearn.school.blog/).
## Dependencies
This implementation uses Python, Keras with Theano backend, and Scikit Learn. 

## Dataset 
Please extract the contents from training_and_testing_data_corrnet.rar file and keep it in the same folder as the DeepLearn_corrnet.py script.

## Usage:
Training and testing the model on MNIST dataset.

```python
>>> model,branchModel = buildModel(loss_type = 2)
>>> trainModel(model, loss_type = 2)
 
Training with following architecture....
L_Type: l2+l3-L4
h_dim: 50
hdim_deep: 500
hdim_deep2: 300
lamda: 0.02
 
Training done....
 
>>> testModel(branchModel)
 
view1 to view2 transfer accuracy
0.8879
view2 to view1 transfer accuracy
0.8964
 
test sum-correlation
49.1316743225
```
Reconstruction of one view given the other
```python
>>> reconstruct_from_left(model,X_train_l[6:7])
```
![Left2right reconstruction](https://cloud.githubusercontent.com/assets/22491381/26366296/2a07d9e6-4008-11e7-9d17-f3708c172f1d.PNG)

```python
>>> reconstruct_from_right(model,X_train_r[6:7])
```
![Right2left reconstruction](https://cloud.githubusercontent.com/assets/22491381/26366297/2a0c6a6a-4008-11e7-8b3e-a55d2bb29988.PNG)

