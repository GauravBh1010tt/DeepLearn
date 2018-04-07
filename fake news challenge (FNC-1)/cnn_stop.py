# -*- coding: utf-8 -*-

from __future__ import print_function
import utility
import warnings
import numpy as np
import gensim as gen
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Merge
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
import pandas as pd
from keras.layers.convolutional import Convolution1D
# set parameters:
from nltk.tokenize import regexp_tokenize
warnings.simplefilter("ignore")

embedding_dim = 300
LSTM_neurons = 50
dense_neuron = 16
dimx = 100
dimy = 200
lamda = 0.0
nb_filter = 100
filter_length = 4
vocab_size = 10000
batch_size = 50
epochs = 5
ntn_out = 16
ntn_in = nb_filter 
state = False

def preprocess_data(head, body):
    stop_words = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
        "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
        "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
        "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
        "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
        "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
        "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
        "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
        "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
        "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
        "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
        "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
        "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
        ]
    chead, cbody = [],[]
    for sample in head:
        sentence = ' '.join([word for word in sample.split() if word not in stop_words])
        chead.append(sentence)
      
    for sample in body:
        sentence = ' '.join([word for word in sample.split() if word not in stop_words])
        cbody.append(sentence)
    print(cbody[0])
    return chead, cbody

def generateMatrix(obj, sent_Q, sent_A, dimx, dimy):
    START = '$_START_$'
    END = '$_END_$'
    unk_token = '$_UNK_$'
    sent1 = []
    #sent1_Q = ques_sent
    #sent1_A = ans_sent
    sent1.extend(sent_Q)
    #sent.extend(ques_sent)
    sent1.extend(sent_A)
    #sent1 = [' '.join(i) for i in sent1]
    #sent.extend(ans_sent)
    sentence = ["%s %s %s" % (START,x,END) for x in sent1]
    tokenize_sent = [regexp_tokenize(x, 
                                     pattern = '\w+|$[\d\.]+|\S+') for x in sentence]
                        
    #for i in index_to_word1:
    #    index_to_word.append(i)
    # for key in word_to_index1.keys():
    #    word_to_index[key] = word_to_index1[key]
        
    for i,sent in enumerate(tokenize_sent):
        tokenize_sent[i] = [w if w in obj.word_to_index else unk_token for w in sent]
        
    len_train = len(sent_Q)
    text=[]
    for i in tokenize_sent:
        text.extend(i)
        
    sentences_x = []
    sentences_y = []
        
        #print 'here' 
        
    for sent in tokenize_sent[0:len_train]:
        temp = [START for i in range(dimx)]
        for ind,word in enumerate(sent[0:dimx]):
            temp[ind] = word
        sentences_x.append(temp)
            
    for sent in tokenize_sent[len_train:]:
        temp = [START for i in range(dimy)]
        for ind,word in enumerate(sent[0:dimy]):
            temp[ind] = word       
        sentences_y.append(temp)
            
    X_data = []
    for i in sentences_x:
        temp = []
        for j in i:
            temp.append(obj.word_to_index[j])
        temp = np.array(temp).T
        X_data.append(temp)
        
    y_data=[]
    for i in sentences_y:
        temp = []
        for j in i:
            temp.append(obj.word_to_index[j])
        temp = np.array(temp).T
        y_data.append(temp)
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    return X_data, y_data


def word2vec_embedding_layer(embedding_matrix):
    #weights = np.load('Word2Vec_QA.syn0.npy')
    layer = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix])
    return layer
'''
try:
    word = wordVec_model['word']
    print('using loaded model.....')
except:
    wordVec_model = gen.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz",binary=True)
#bre'''
file_head = "/fncdata/train_stances.csv" 
file_body = "/fncdata/train_bodies.csv"
head = pd.read_csv(file_head)
body = pd.read_csv(file_body)
head_array = head.values
body_array = body.values                    ##########
print(len(head_array))
print(len(body_array))
labels = head_array[:,2]
body_id = head_array[:,1]
dataset_headLines = head_array[:,0]         ##########
body_ds = []
for i in range(len(head_array)):
    for j in range(len(body_array)):
        if body_array[j][0] == body_id[i]:
            body_ds.append(body_array[j][1])
            break

dataset_body = np.array(body_ds)
#print(type(dataset_body))
new_lab = []
for i in labels:
    if i == 'unrelated':
        new_lab.append(3)
    if i == 'agree':
        new_lab.append(0)
    if i == 'discuss':
        new_lab.append(2)
    if i == 'disagree':
        new_lab.append(1)
y_train = np.array(new_lab)

print("Refining training dataset for CNN")
train_rdh = []
for i in dataset_headLines:
    sentence = ""
    for char in i:
        if char.isalpha() or char == ' ':
            sentence+=char.lower()
        else:
            sentence+=' '
    train_rdh.append(sentence)

train_rdb = []
for i in dataset_body:
    sentence = ""
    for char in i:
        if char.isalpha() or char == ' ':
            sentence+=char.lower()
        else:
            sentence+=' '
    train_rdb.append(sentence)

print("Preprocessing train dataset")
train_rpdh, train_rpdb = preprocess_data(train_rdh, train_rdb)


obj = utility.sample()
train_head,train_body,embedding_matrix = obj.process_data(sent_Q=train_rdh,
                                                     sent_A=train_rdb,dimx=dimx,dimy=dimy,
                                                     wordVec_model = None)

#def buildModel():
inpx = Input(shape=(dimx,),dtype='int32',name='inpx')
x = Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=dimx)(inpx)
#x = word2vec_embedding_layer(embedding_matrix)(inpx)  
inpy = Input(shape=(dimy,),dtype='int32',name='inpy')
y = Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=dimy)(inpy)
#y = word2vec_embedding_layer(embedding_matrix)(inpy)
ques = Convolution1D(nb_filter=nb_filter, filter_length=filter_length,
                     border_mode='valid', activation='relu',
                     subsample_length=1)(x)
                        
ans = Convolution1D(nb_filter=nb_filter, filter_length=filter_length,
                    border_mode='valid', activation='relu',
                    subsample_length=1)(y)
        
#hx = Lambda(max_1d, output_shape=(nb_filter,))(ques)
#hy = Lambda(max_1d, output_shape=(nb_filter,))(ans)
hx = GlobalMaxPooling1D()(ques)
hy = GlobalMaxPooling1D()(ans)

#wordVec_model = []
h =  Merge(mode="concat",name='h')([hx,hy])
#h = NeuralTensorLayer(output_dim=1,input_dim=ntn_in)([hx,hy])
#h = ntn_layer(ntn_in,ntn_out,activation=None)([hx,hy])
#score = h
wrap = Dense(dense_neuron, activation='relu',name='wrap')(h)
#score = Dense(1,activation='sigmoid',name='score')(h)
#wrap = Dense(dense_neuron,activation='relu',name='wrap')(h)
score = Dense(4,activation='softmax',name='score')(wrap)

#score=K.clip(score,1e-7,1.0-1e-7)
#corr = CorrelationRegularization(-lamda)([hx,hy])
#model = Model([inpx,inpy],[score,corr])
model = Model([inpx,inpy],score)
model.compile(loss='categorical_crossentropy',optimizer="adadelta",metrics=['accuracy'])
    

print('data split')
Y_train = to_categorical(y_train, 4)
#train_head_split, test_head_split, train_body_split, test_body_split, train_y_split, test_y_split = utility.split(train_head, train_body, Y_train, 0.2)
model.fit([train_head,train_body],Y_train, nb_epoch = 10, verbose=2)

file0 = "/fncdata/competition_test_stances.csv"
file1 = "/fncdata/test_bodies.csv"
test_head = pd.read_csv(file0)
test_body = pd.read_csv(file1)
test_head = test_head.values
test_body = test_body.values
test_hds = test_head[:,0]
test_ids = test_head[:,1]
test_labels = test_head[:,2]
test_bds = []
for ids in test_ids:
    for body in test_body:
        if ids == body[0]:
            test_bds.append(body[1])

print("refining test dataset")
test_rdh = []
for i in range(len(test_hds)):
    sentence = ""
    for char in test_hds[i]:
        if char.isalpha() or char == ' ':
            sentence+=char.lower()
        else:
            sentence+=' '
    test_rdh.append(sentence)

test_rdb = []
for i in range(len(test_bds)):
    sentence = ""
    for char in test_bds[i]:
        if char.isalpha() or char == ' ':
            sentence+=char.lower()
        else:
            sentence+=' '
    test_rdb.append(sentence)

print("Preprocessing test dataset")
test_rpdh, test_rpdb = preprocess_data(test_rdh, test_rdb)
ts_head, ts_body = generateMatrix(obj, test_rdh, test_rdb, dimx, dimy)
predictions = model.predict([ts_head, ts_body])
predictions = [i.argmax()for i in predictions]
predictions = np.array(predictions)
string_predicted = []
for i,j in enumerate(predictions):
    if j == 3:
        string_predicted.append("unrelated")
    elif j == 0:
        string_predicted.append("agree")
    elif j == 1:
        string_predicted.append("disagree")
    elif j == 2:
        string_predicted.append("discuss")
        
from sklearn.metrics import accuracy_score
from utils.score import report_score
score = accuracy_score(string_predicted, test_labels)
print(("Accuracy on test dataset: ",score))
report_score(string_predicted, test_labels)