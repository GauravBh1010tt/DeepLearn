# -*- coding: utf-8 -*-

from __future__ import print_function
from fnc_libs import *
from keras import optimizers as op
from util import *
import random
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Layer, Dropout, Activation, Input, Merge, Multiply
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
from keras import regularizers

from utils.score import report_score, LABELS, score_submission


# Set file names
file_train_instances = "/fncdata/train_stances.csv"
file_train_bodies = "/fncdata/train_bodies.csv"
file_test_instances = "/fncdata/competition_test_stances.csv"
file_test_bodies = "/fncdata/test_bodies.csv"

file_head = "/fncdata/competition_test_stances.csv"
file_body = "/fncdata/test_bodies.csv"

test_dh,test_db,test_stances = load_data(file_head,file_body)

#file_test_stances = "labeled_ts.csv"
#dataa = pd.read_csv(file_test_stances)
#test_ = dataa.values
#test_stances = test_[:,2]

new_test_stances = []
for i in test_stances:
    if i == 'unrelated':
        new_test_stances.append(3)
    elif i == 'agree':
        new_test_stances.append(0)
    elif i == 'disagree':
        new_test_stances.append(1)
    else:
        new_test_stances.append(2)

# Initialise hyperparameters
r = random.Random()
lim_unigram = 5000
target_size = 4
hidden_size = 100
train_keep_prob = 0.6
l2_alpha = 0.00001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 500
epochs = 90
reg = 0.00005
# Load data sets
raw_train = FNCData(file_train_instances, file_train_bodies)
raw_test = FNCData(file_test_instances, file_test_bodies)
n_train = len(raw_train.instances)

print ("processing data .....")
# Process data sets
train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
    pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
feature_size = len(train_set[0])
test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

model = Sequential()
print("creating model")
model.add(Dense(500, input_dim=10001, activation='relu',\
                kernel_regularizer=regularizers.l2(reg)))
                #activity_regularizer=regularizers.l1(0.001)))
#model.add(Dropout(0.4))
#model.add(Dense(100, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(100, activation='relu',\
                kernel_regularizer=regularizers.l2(reg)))
                #activity_regularizer=regularizers.l1(0.001)))
model.add(Dropout(0.4))
model.add(Dense(4, activation='softmax'))
print("compiling")

opt = op.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=5.)

model.compile( loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
Y_train = to_categorical(train_stances, 4)
Y_test = to_categorical(new_test_stances, 4)
train_set = np.array(train_set)
test_set = np.array(test_set)
print("fitting")
model.fit(train_set,Y_train, batch_size=500, epochs=90, verbose=0)
predictions = model.predict(test_set)
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

report_score(test_stances, string_predicted)

'''
# Define model
# Create placeholders
features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
stances_pl = tf.placeholder(tf.int64, [None], 'stances')
keep_prob_pl = tf.placeholder(tf.float32)

# Infer batch size
batch_size = tf.shape(features_pl)[0]

# Define multi-layer perceptron
hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), keep_prob=keep_prob_pl)
logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size), keep_prob=keep_prob_pl)
logits = tf.reshape(logits_flat, [batch_size, target_size])

# Define L2 loss
tf_vars = tf.trainable_variables()
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

# Define overall loss
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, stances_pl) + l2_loss)

# Define prediction
softmaxed_logits = tf.nn.softmax(logits)
predict = tf.arg_max(softmaxed_logits, 1)


# Load model
if mode == 'load':
    with tf.Session() as sess:
        load_model(sess)


        # Predict
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)


# Train model
if mode == 'train':

    # Define optimiser
    opt_func = tf.train.AdamOptimizer(learn_rate)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
    opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

    # Perform training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            total_loss = 0
            indices = list(range(n_train))
            r.shuffle(indices)

            for i in range(n_train // batch_size_train):
                batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
                batch_features = [train_set[i] for i in batch_indices]
                batch_stances = [train_stances[i] for i in batch_indices]

                batch_feed_dict = {features_pl: batch_features, stances_pl: batch_stances, keep_prob_pl: train_keep_prob}
                _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
                total_loss += current_loss


        # Predict
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)


# Save predictions
save_predictions(test_pred, file_predictions)
'''