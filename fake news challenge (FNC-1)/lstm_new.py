# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from fnc_libs import *

d = DataSet()
folds,hold_out = kfold_split(d,n_folds=10)
fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

wordVec_model = gen.models.KeyedVectors.load_word2vec_format("/fncdata1/GoogleNews-vectors-negative300.bin.gz",binary=True)

filename = "/fncdata/train_bodies.csv"

body = pd.read_csv(filename)
body_array = body.values
train_dh = []
train_db = []
train_ds = []

print("Generating train dataset for CNN")
for i in range(len(fold_stances)):
    for j in range(len(fold_stances[i])):
        train_dh.append(fold_stances[i][j]["Headline"])
        train_ds.append(fold_stances[i][j]["Stance"])

for i in range(len(fold_stances)):
    for j in range(len(fold_stances[i])):
        body_id = fold_stances[i][j]["Body ID"]
        for m in range(len(body_array)):
            if body_id == body_array[m][0]:
                train_db.append(body_array[m][1])

print("Refining training dataset for CNN")
train_rdh = []
for i in range(len(train_dh)):
    sentence = ""
    for char in train_dh[i]:
        if char.isalpha() or char == ' ':
            sentence+=char.lower()
        else:
            sentence+=' '
    train_rdh.append(sentence)

train_rdb = []
for i in range(len(train_db)):
    sentence = ""
    for char in train_db[i]:
        if char.isalpha() or char == ' ':
            sentence+=char.lower()
        else:
            sentence+=' '
    train_rdb.append(sentence)
train_rds = []

for i,j in enumerate(train_ds):
    if j == "unrelated":
        train_rds.append("2")
    elif j == "agree":
        train_rds.append("1")
    elif j == "disagree":
        train_rds.append("0")
    elif j == "discuss":
        train_rds.append("3")
            
print("Generating test dataset for CNN")
'''
test_dh, test_db, test_ds = [],[],[]

for i in range(len(hold_out_stances)):
    test_dh.append(hold_out_stances[i]["Headline"])
    test_ds.append(hold_out_stances[i]["Stance"])
    

for i in range(len(hold_out_stances)):
    body_id = hold_out_stances[i]["Body ID"]
    for m in range(len(body_array)):
        if body_id == body_array[m][0]:
            test_db.append(body_array[m][1])
'''
         
file_head = "/fncdata/competition_test_stances.csv"
file_body = "/fncdata/test_bodies.csv"

test_dh,test_db,test_ds = load_data(file_head,file_body)

print("Refining testing dataset for CNN")
test_rdh = []
for i in range(len(test_dh)):
    sentence = ""
    for char in test_dh[i]:
        if char.isalpha() or char == ' ':
            sentence+=char.lower()
        else:
            sentence+=' '
    test_rdh.append(sentence)

test_rdb = []
for i in range(len(test_db)):
    sentence = ""
    for char in test_db[i]:
        if char.isalpha() or char == ' ':
            sentence+=char.lower()
        else:
            sentence+=' '
    test_rdb.append(sentence)    

obj = utility.sample()

print("Training CNN")
model,tr_head,tr_body = trainCNN(obj, wordVec_model, train_rdh,train_rdb)
ts_head, ts_body = generateMatrix(obj,test_rdh, test_rdb)
Y_train = to_categorical(train_rds, 4)
model.fit([tr_head,tr_body],Y_train, nb_epoch = 10, verbose=2)

print ('\n model trained....\n')

predictions = model.predict([ts_head, ts_body])
predictions = [i.argmax()for i in predictions]
predictions = np.array(predictions)
string_predicted = []
for i,j in enumerate(predictions):
    if j == 2:
        string_predicted.append("unrelated")
    elif j == 1:
        string_predicted.append("agree")
    elif j == 0:
        string_predicted.append("disagree")
    elif j == 3:
        string_predicted.append("discuss")

import sklearn
score = sklearn.metrics.accuracy_score(test_ds,string_predicted)
report_score(test_ds, string_predicted)