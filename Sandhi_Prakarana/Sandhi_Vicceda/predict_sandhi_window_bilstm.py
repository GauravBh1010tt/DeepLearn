from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Embedding, Reshape, Dropout
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import train_test_data_prepare as sdp

def train_predict_sandhi_window(dtrain, dtest, mode):
    batch_size = 64  # Batch size for training.
    epochs = 40  # Number of epochs to train for.
    latent_dim = 64  # Latent dimensionality of the encoding space.
    inwordlen = 5

    # Vectorize the data.
    inputs = []
    targets = []
    characters = set()
    
    for data in dtrain:
        target = np.zeros(len(data[3]))
        input_word = data[3]
    
        inputs.append(input_word)
        for i in range(data[4], data[5]):
            target[i] = 1
        targets.append(target)
    
        for char in input_word:
            if char not in characters:
                characters.add(char)

    maxlen = max([len(s) for s in inputs])
    print(maxlen)

    """
    * is used as padding character
    """
    characters.add('*')
    char2idx = dict([(char, i) for i, char in enumerate(characters)])
    num_tokens = len(characters)
    
    X_train = [[char2idx[c] for c in w] for w in inputs]
    X_train = pad_sequences(maxlen=maxlen, sequences=X_train, padding="post", value=char2idx['*'])
    
    Y_train = targets
    Y_train = pad_sequences(maxlen=maxlen, sequences=Y_train, padding="post", value=0.0)
    Y_train = np.array(Y_train).reshape(-1, maxlen, 1)
    
    inputs = []
    targets = []
    for data in dtest:
        target = np.zeros(len(data[3]))
        input_word = data[3]
    
        inputs.append(input_word)
        for i in range(data[4], data[5]):
            target[i] = 1
        targets.append(target)
    
        for char in input_word:
            if char not in characters:
                characters.add(char)
    
    X_test = [[char2idx[c] for c in w] for w in inputs]
    X_test = pad_sequences(maxlen=maxlen, sequences=X_test, padding="post", value=char2idx['*'])
    
    Y_test = targets
    Y_test = pad_sequences(maxlen=maxlen, sequences=Y_test, padding="post", value=0.0)
    Y_test = np.array(Y_test).reshape(-1, maxlen, 1)
    
    print('Number of training samples:', len(X_train))
    print('Number of unique tokens:', num_tokens)
    
    # Define an input sequence and process it.
    inputword = Input(shape=(maxlen,))
    embed = Embedding(input_dim=num_tokens, output_dim=8, input_length=maxlen, mask_zero=True)(inputword)
    bilstm = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
    out, forward_h, forward_c, backward_h, backward_c = bilstm(embed)
    outd = Dropout(0.5)(out)
    outputtarget = Dense(1, activation="sigmoid")(outd)
    
    model = Model(inputword, outputtarget)
    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
    #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, Y_train, batch_size, epochs, validation_split=0.1)
    
    # Save model and test files
    #model.save('bilstm.h5')
    #np.save('testX', X_test)
    #np.save('testY', Y_test)
    
    np.set_printoptions(precision=2, suppress=True)
    passed = 0
    failed = 0
   
    startlist = []
    for i in range(X_test.shape[0]):
        test = X_test[i].reshape((-1, maxlen))
        res = model.predict(test)
        res = res.reshape((maxlen))
        dup = np.copy(res)
        act = Y_test[i].reshape((maxlen))
    
        maxsum = 0
        maxstart = 0
        for i in range(maxlen-inwordlen):
            sumword = 0
            for j in range(inwordlen):
                sumword = sumword + dup[i+j]
            if maxsum < sumword:
                maxsum = sumword
                maxstart = i

        startlist.append(maxstart)

        if mode == 0:
            for k in range(len(act)):
                if act[k] == 1:
                    break
    
            if k == maxstart:
                passed = passed + 1
            else:
                failed = failed + 1
            """
                print(act)
                print(dup)
                print("****************************************************")
            """
    if mode == 0:
        print(passed)
        print(failed)
        print(passed*100/(passed+failed))

    return startlist

#dl = sdp.get_xy_data("../sandhi/Data/sandhiset.txt")
#dtrain, dtest = train_test_split(dl, test_size=0.2, random_state=1)
#train_predict_sandhi_window(dtrain, dtest, 0)
