"""
** deeplean-ai.com **
** dl-lab **
created by :: GauravBh1010tt
"""

import string

from nltk.corpus import words
from textstat.textstat import textstat
from nltk import pos_tag, word_tokenize
from nltk.stem.lancaster import LancasterStemmer

LanStem = LancasterStemmer()

#Average Character Per Word In A Sentence(Returns Float):
def CPW(text):
    count_char = 0
    count_word = 0.0
    for word in text.split():
        count_word += 1.0
        count_char += len(word.strip(string.punctuation))
    return count_char/count_word

#Number Of Words Per Sentence(Returns Integer):
def WPS(text):
    count = 0
    for word in text.split():
        if word in set(w.lower() for w in words.words()):
            count += 1
    return count

#Average Number Of Syllables In Sentence(Returns Float):
def SPW(text):
    count = 0
#    text = tokenize(text)
    text = text.split()
    vowels = 'a e i o u y'
    for word in text:
        if word not in ", . ? ! : # $ % ^ & * ( ) { } [ ]".split():
            if word[0] in vowels.split():
                count +=1
                for index in range(1,len(word)):
                    if word[index] in vowels and word[index-1] not in vowels:
                        count +=1
                    if word.endswith('e'):
                        count -= 1
                    if word.endswith('le'):
                        count+=1
                    if count == 0:
                        count +=1
    return float(count)/len(text)

#Long Words In A Sentence(Returns Integer):
def LWPS(text):
    text = text.split()
    count = 0
    for word in text:
        if len(word) > 7:
            count += 1
    return count

#Fraction Of Long Words In A Sentence(Returns Float):
def LWR(text):
    text = text.split()
    count = 0
    for word in text:
        if len(word) > 7:
            count += 1
    return float(count)/len(text)

#Number Of Complex Word Per Sentence(Returns Float):
def CWPS(text):
    count = 0
    for word in text.split():
        if(LanStem.stem(word) != word):
            count += 1
    return float(count)/len(text.split())

#Dale-Chall Readability Index(Returns Float):
def DaleChall(text):
    return textstat.dale_chall_readability_score(text)

#Edit Distance Value For Two String(Returns Integer):
def ED(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

#Get A List Of Nouns From String(Returns List Of Sting):
def nouns(text):
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)]
    return nouns

#Average Edit Distance Value For Two String And The Average Edit Distance Between The Nouns Present In Them(Returns Float)
def EditDist_Dist(t1,t2):
    tot = 0
    for w1 in t1.split():
        for w2 in t2.split():
            tot += ED(w1, w2)
    return float(tot)/(len(t1.split()) * len(t2.split()))

def EditDist_Noun(sent_A, sent_B):
    tot = 0
    n1 = nouns(sent_A)
    n2 = nouns(sent_B)
    for w1 in n1:
        for w2 in n2:
            tot += ED(w1, w2)
    temp1 = len(n1)
    temp2 = len(n2)
    if len(n1) == 0:
        temp1 = 1
    if len(n2) == 0:
        temp2 = 1
    return float(tot)/(temp1 * temp2)

#Longest Common Subsequence(Returns Integer):
def LCS_Len(a, b):
    a = a.lower()
    b = b.lower()
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert a[x-1] == b[y-1]
            result = a[x-1] + result
            x -= 1
            y -= 1
    return len(result)

#Length Of Longest Common Subsequence(Returns Integer):
def LCW(t1, t2):
    lcs = []
    for w1 in t1.split():
        for w2 in t2.split():
            if w1 in w2:
                lcs.append(len(w1))
    if len(lcs)==0:
        return 0
    else:
        return max(lcs)