
# CorrNet

This is an implementation of Correlational Neural Network (CorrNet) described in the following paper : *Sarath Chandar, Mitesh M Khapra, Hugo Larochelle, Balaraman Ravindran. [Correlational Neural Networks](https://arxiv.org/pdf/1504.07225.pdf)*. 

![CorrNet](https://cloud.githubusercontent.com/assets/22491381/26366765/e31809d2-4009-11e7-80e2-d79cfd04a418.PNG)

### For detailed description please refer to my blog post [COMMON REPRESENTATION LEARNING USING DEEP CORRNET](http://deeplearn-ai.com/2017/05/24/common-representation-learning-using-deep-corrnet/).
## Dependencies
This implementation uses Python 2.7, Keras (2.0 or above) with Theano backend, and Scikit Learn. 

## Dataset 
Please extract the contents from **training_and_testing_data_corrnet.rar** file and keep it in the same folder as the DeepLearn_corrnet.py script.

## Usage:
Training and testing the model on MNIST dataset.

```python
>>> left_view, right_view = prepare_data()
>>> model,branchModel = buildModel(loss_type)
>>> trainModel(model,left_view,right_view,loss_type=2,nb_epoch=40,batch_size=100)
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
>>> reconstruct_from_left(model,left_view[6:7])
```
![Left2right reconstruction](https://i0.wp.com/deeplearnschool.files.wordpress.com/2017/05/git1.png?ssl=1&w=450)

```python
>>> reconstruct_from_right(model,right_view[6:7])
```
![Right2left reconstruction](https://i0.wp.com/deeplearnschool.files.wordpress.com/2017/05/git2.png?ssl=1&w=450)

