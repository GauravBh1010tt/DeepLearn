from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate
import numpy as np
from sklearn.model_selection import train_test_split
import train_test_data_prepare as sdp


def train_sandhi_split(dtrain, dtest, mode):
    batch_size = 64  # Batch size for training.
    epochs = 30  # Number of epochs to train for.
    latent_dim = 128  # Latent dimensionality of the encoding space.
    inwordlen = 5

    # Vectorize the data.
    input_texts = []
    target_texts = []
    X_tests = []
    Y_tests = []
    characters = set()

    for data in dtrain:
        target_text = data[0] + '+' + data[1]
        input_text = data[2]
    
        # We use "&" as the "start sequence" character for the targets, and "$" as "end sequence" character.
        target_text = '&' + target_text + '$'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in characters:
                characters.add(char)
        for char in target_text:
            if char not in characters:
                characters.add(char)
    
    for data in dtest:
        target_text = data[0] + '+' + data[1]
        input_text = data[2]

        # We use "&" as the "start sequence" character for the targets, and "$" as "end sequence" character.
        target_text = '&' + target_text + '$'
        X_tests.append(input_text)
        Y_tests.append(target_text)
        for char in input_text:
            if char not in characters:
                characters.add(char)
        for char in target_text:
            if char not in characters:
                characters.add(char)
    
    # Using '*' for padding 
    characters.add('*')
    
    characters = sorted(list(characters))
    num_tokens = len(characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    
    print('Number of samples:', len(input_texts))
    print('Number of unique tokens:', num_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    
    token_index = dict([(char, i) for i, char in enumerate(characters)])
    
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_tokens), dtype='float32')
    
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, token_index[char]] = 1.
        encoder_input_data[i, t + 1:, token_index['*']] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, token_index[char]] = 1.
        decoder_input_data[i, t + 1:, token_index['*']] = 1.
        decoder_target_data[i, t:, token_index['*']] = 1.
    
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_tokens))
    encoder = Bidirectional(LSTM(latent_dim, return_state=True, dropout=0.5))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_tokens))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True, dropout=0.5)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1)
    
    # Save model
    model.save('bis2s.h5')
    
    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)
    
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, token_index['&']] = 1.
    
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)
    
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char
    
            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '$' or
               len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True
    
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
    
            # Update states
            states_value = [h, c]
    
        return decoded_sentence
    
    input_texts = []
    target_texts = []
    
    input_texts = X_tests
    target_texts = Y_tests

    for i in range(len(target_texts)):
        target_texts[i] = target_texts[i][1:-1]
    
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_tokens), dtype='float32')
    
    for i, input_text in enumerate(input_texts):
        for t, char in enumerate(input_text):
            if char not in token_index:
                continue
            encoder_input_data[i, t, token_index[char]] = 1.
        encoder_input_data[i, t + 1:, token_index['*']] = 1.
    
    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states
    
    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = Input(shape=(latent_dim*2,))
    decoder_state_input_c = Input(shape=(latent_dim*2,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    # Reverse-lookup token index to decode sequences back to something readable.
    reverse_input_char_index = dict((i, char) for char, i in token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in token_index.items())
    
    total = len(encoder_input_data)
    passed = 0
    results = []
    for seq_index in range(len(encoder_input_data)):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        decoded_sentence = decoded_sentence.strip()
        decoded_sentence = decoded_sentence.strip('$')
        results.append(decoded_sentence)

        if mode == 0:
            if decoded_sentence == target_texts[seq_index]:
                passed = passed + 1
            """
            else:
                print(str(seq_index)+'/'+str(total))
                print('-')
                print('Input sentence:   ', input_texts[seq_index])
                print('Decoded sentence: ', decoded_sentence)
                print('Expected sentence:', target_texts[seq_index])
            """
    if mode == 0:
        print("Passed: "+str(passed)+'/'+str(total)+', '+str(passed*100/total))

    return results

#dl = sdp.get_xy_data("../sandhi/Data/sandhiset.txt")
#dtrain, dtest = train_test_split(dl, test_size=0.2, random_state=1)
#train_sandhi_split(dtrain, dtest, 0)
