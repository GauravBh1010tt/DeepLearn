"""
** deeplean-ai.com **
** dl-lab **
created by :: GauravBh1010tt
"""

import pandas as pd
import numpy as np
import re

#from tqdm import tqdm
from nltk.corpus import wordnet
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine as cos
from stop_words import get_stop_words
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

#Number Of Words In A String(Returns Integer):
def length(val):
    return len(val.split())

#Whether A String Is Subset Of Other(Returns 1 and 0):
def substringCheck(sen_A, sen_B):
    if sen_A in sen_B or sen_B in sen_A:
        return 1
    else:
        return 0

#Number Of Same Words In Two Sentences(Returns Float):
def overlap(sen_A, sen_B):
     a = sen_A.split()
     b = sen_B.split()
     count = 0
     for word_a in a:
         for word_b in b:
             if(word_a == word_b):
                 count += 1
     return count

#Number Of Synonyms In Two Sentences(Returns Float):
def overlapSyn(sen_A, sen_B):
    a = sen_A.split()
    b = sen_B.split()
    word_synonyms = []
    for word in a:
        for synset in wordnet.synsets(word):
            for lemma in synset.lemma_names():
                if lemma in b and lemma != word:
                    word_synonyms.append(lemma)
    return len(word_synonyms)

#Forming Bag Of Words[BOW][Returns BOW Dictionary]:
def train_BOW(lst):
    temp = []
    for sent in lst:
        temp.extend(sent.split())
    counts = Counter(temp)
    total_count = len(set(temp))
    for word in counts:
        counts[word] /= float(total_count)
    return counts
        
#Sum Of BOW Values For A Sent[Returns Float]:
def Sum_BOW(sent, dic):
    tot = 0.0
    for word in sent.split():
        try:
            tot += dic[word]
        except:
            continue
    return tot

#Training Bigram Model[Returns Dictionary of Dictionaries]:
def train_bigram(lst):
    model = defaultdict(lambda: defaultdict(lambda: 0))

    for sent in lst:
        sent = sent.split()
        for w1, w2 in bigrams(sent, pad_right=True, pad_left=True):
            model[w1][w2] += 1  
    total_count = 0      
    for w1 in model:
        total_count = float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2] /= total_count
    return model

#Total Sum Of Bigram Probablity Of A Sentence[Returns Float]:
def sum_bigram(sent, model):
    sent = sent.split()
    first = True
    tot = 0
    for i in range(len(sent)):
        try:
            if first:
                tot += model[None][sent[i]]
                first = False
            else:
                tot += model[sent[i-1]][sent[i]]
        except:
            continue
    return tot

#Training Trigram Model[Returns Dictionary of Dictionaries]:
def train_trigram(lst):
    model = defaultdict(lambda: defaultdict(lambda: 0))
    for sent in lst:
        sent = sent.split()
        for w1, w2, w3 in trigrams(sent, pad_right=True, pad_left=True):
            model[(w1,w2)][w2] += 1
    total_count = 0
    for w1,w2 in model:
        total_count = float(sum(model[(w1, w2)].values()))
        for w3 in model[(w1,w2)]:
            model[(w1, w2)][w3] /= total_count

#Total Sum Of Trigram Probablity Of A Sentence[Returns Float]:
def sum_trigram(sent, model):
    sent = sent.split()
    first = True
    second = True
    tot = 0
    for i in range(len(sent)):
        try:
            if first:
                tot += model[None, None][sent[i]]
                first = False
            elif second:
                tot += model[None, sent[i-1]][sent[i]]
                second = False
            else:
                tot += model[sent[i-2], sent[i-1]][sent[i]]
        except:
            continue
    return tot

#Word2Vec Training(Returns Vector):
def W2V_train(lst1, lst2):
    vocab = []
    for i in range(len(lst1)):
        w1 = lst1[i]
        w2 = lst2[i]
        vocab.append(w1.split())
        vocab.append(w2.split())
    for temp in vocab:
        for j in range(len(temp)):
            temp[j] = temp[j].lower()
    return Word2Vec(vocab)

#Returns The Difference Between Word2Vec Sum Of All The Words In Two Sentences(Returns Vec):
def W2V_Vec(sent_A, sent_B, vec):
    if len(sent_A) <= 1:
        sent_A += 'none'

    elif len(sent_B) <= 1:
        sent_B += 'none'
    vec1 = 0
    vec2 = 0
    sent_A = tokenize(sent_A)
    sent_B = tokenize(sent_B)
    
    for word in sent_A:
        if word not in ", . ? ! # $ % ^ & * ( ) { } [ ]".split():
            try:
                vec1 += vec[word]
            except:
                continue 
    for word in sent_B:
        if word not in ", . ? ! # $ % ^ & * ( ) { } [ ]".split():
            try:
                vec2 += vec[word]
            except:
                continue
    try:
        result = cos(vec1, vec2)
    except:
        result = 0.0
    
    if np.isnan(result):
        return 0.0
    else:
        return result

#Trains LDA Model (Returns Model):
def LDA_train(doc):
    red = []
    en_stop = get_stop_words('en')
    for d in doc:
        try:
            raw = d.lower()
            tokens = tokenizer.tokenize(raw)
            stopped_tokens = [i for i in tokens if not i in en_stop]
            red.append(stopped_tokens)
        except:
            continue
    print("Forming Dictionary.....")
    dictionary = corpora.Dictionary(red)
    print("Forming Corpus.....")
    corpus = [dictionary.doc2bow(text) for text in red]
    print("Training Model.....")
    lda = models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=1)
    return lda

#Returns Average Of Probablity Of Word Present In LDA Model For Input Document(Returns Float):
def LDA(doc1, doc2, lda):
    word = pd.DataFrame()
    weight = pd.DataFrame()
    vec1 = []
    vec2 = []
    for i in range(10):
        vec1.append(0)
        vec2.append(0)
    
    for i in range(10):
        a = []
        wrd = []
        wgt = []
        for x in lda.print_topic(i).split():
            if x != '+':
                a.append(x)
        for w in a:
            t = w.split("*")
            wrd.append(t[1][1:-1])
            wgt.append(float(t[0]))
        word[i] = wrd
        weight[i] = wgt
    num = 0
    wrd1 = []
    wrd2 = []

#    print 'Vector Formation for doc1.....'
    
    for d in doc1.split():
        for i in range(10):
            for j in range(10):
                if d.lower() == word[i][j]:
                    vec1[j] += float(weight[i][j])
                    wrd1.append(word[i][j])
    
#    print 'Vector Formation for doc2.....'
    
    for d in doc2.split():
        for i in range(10):
            for j in range(10):
                if d.lower() == word[i][j]:
                    vec2[i] += float(weight[i][j])
                    wrd2.append(word[i][j])
    v1 = 0.0
    v2 = 0.0
    for i in range(10):
        if vec1[i] >= v1:
            t1 = i
            v1 = vec1[i]
        if vec2[i] >= v2:
            t2 = i
            v2 = vec2[i]
    wrd1_list = list(set(wrd1))
    wrd2_list = list(set(wrd2))
    w1_len = len(wrd1_list)
    w2_len = len(wrd2_list)
    w1_new = []
    w2_new = []
    for i in range(w1_len):
        d = wrd1_list[i]
        for i in range(10):
            if d != word[t2][i]:
                w1_new.append(d)
    for i in range(w2_len):
        d = wrd2_list[i]
        for i in range(10):
            if d != word[t1][i]:
                w2_new.append(d)
    num = len(list(set(w1_new))) + len(set(w2_new))
    try:
        return num
    except:
        return 0.0
