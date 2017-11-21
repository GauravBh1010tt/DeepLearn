import ntn_input
import ntn
import scipy.io as sio
import numpy as np

data_number = 0 #0 - Wordnet, 1 - Freebase
if data_number == 0: data_name = 'Wordnet'
else: data_name = 'Freebase'
embedding_size = 100

data_path = 'data\\'+data_name
output_path = 'data\\output\\'+data_name+'\\'

entities_string='/entities.txt'
relations_string='/relations.txt'
embeds_string='/initEmbed.mat'
training_string='/train.txt'
test_string='/test.txt'
dev_string='/dev.txt'

def load_entities(data_path=data_path):
    entities_file = open(data_path+entities_string)
    entities_list = entities_file.read().strip().split('\n')
    entities_file.close()
    return entities_list

def load_relations(data_path=data_path):
    relations_file = open(data_path+relations_string)
    relations_list = relations_file.read().strip().split('\n')
    relations_file.close()
    return relations_list
    
def load_init_embeds(data_path=data_path):
    embeds_path = data_path+embeds_string
    return load_embeds(embeds_path)

def load_embeds(file_path):
    mat_contents = sio.loadmat(file_path)
    words = mat_contents['words']
    we = mat_contents['We']
    tree = mat_contents['tree']
    word_vecs = [[we[j][i] for j in range(embedding_size)] for i in range(len(words[0]))]
    entity_words = [map(int, tree[i][0][0][0][0][0]) for i in range(len(tree))]
    return (word_vecs,entity_words)

def load_training_data(data_path=data_path):
    training_file = open(data_path+training_string)
    training_data = [line.split('\t') for line in training_file.read().strip().split('\n')]
    return np.array(training_data)

def load_dev_data(data_path=data_path):
    #print data_path+dev_string
    dev_file = open(data_path+dev_string)
    dev_data = [line.split('\t') for line in dev_file.read().strip().split('\n')]
    return np.array(dev_data)

def load_test_data(data_path=data_path):
    test_file = open(data_path+test_string)
    test_data = [line.split('\t') for line in test_file.read().strip().split('\n')]
    return np.array(test_data)


def data_to_indexed(data, entities, relations):
    entity_to_index = {entities[i] : i for i in range(len(entities))}
    relation_to_index = {relations[i] : i for i in range(len(relations))}
    indexed_data = [(entity_to_index[data[i][0]], relation_to_index[data[i][1]],\
            entity_to_index[data[i][2]], float(data[i][3])) for i in range(len(data))]
    return indexed_data

#dataset is in the form (e1, R, e2, label)
def data_to_relation_sets(data_batch, num_relations):
    batches = [[] for i in range(num_relations)]
    labels = [[] for i in range(num_relations)]
    for e1,r,e2,label in data_batch:
        batches[r].append((e1,e2,1))
        labels[r].append([label])
    return (batches, labels)

