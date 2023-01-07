# encoding: utf-8
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import devnagri_reader as dr
import numpy as np

swaras = ['a', 'A', 'i', 'I', 'u', 'U', 'e', 'E', 'o', 'O', 'f', 'F', 'x', 'X']
vyanjanas = ['k', 'K', 'g', 'G', 'N', 
             'c', 'C', 'j', 'J', 'Y',
             'w', 'W', 'q', 'Q', 'R',
             't', 'T', 'd', 'D', 'n',
             'p', 'P', 'b', 'B', 'm',
             'y', 'r', 'l', 'v','S', 'z', 's', 'h', 'L', '|']
others = ['H', 'Z', 'V', 'M', '~', '/', '\\', '^', '\'']

slp1charlist = swaras + vyanjanas + others

def remove_nonslp1_chars(word):
    newword = ''
    for char in word:
        if char in slp1charlist:
            newword = newword + char
    return newword

def get_sandhi_dataset(datafile):
    word1list = []
    word2list = []
    outputlist = []

    with open(datafile) as fp:
        tests = fp.read().splitlines()
    
    total = 0
    maxlen = 0

    for test in tests:
        #print(test)
    
        if(test.find('=>') == -1):
            continue;
        inout = test.split('=>')
        words = inout[1].split('+')
    
        if(len(words) != 2):
            continue
    
        word1 = words[0].strip()
        word1 = dr.read_devnagri_text(word1)
        slp1word1 = transliterate(word1, sanscript.DEVANAGARI, sanscript.SLP1)
        slp1word1 = remove_nonslp1_chars(slp1word1)
    
        word2 = words[1].strip()
        word2 = dr.read_devnagri_text(word2)
        slp1word2 = transliterate(word2, sanscript.DEVANAGARI, sanscript.SLP1)
        slp1word2 = remove_nonslp1_chars(slp1word2)
    
        expected = inout[0].strip()
        expected = dr.read_devnagri_text(expected)
        slp1expected = transliterate(expected, sanscript.DEVANAGARI, sanscript.SLP1)
        slp1expected = remove_nonslp1_chars(slp1expected)

        if slp1word1 and slp1word2 and slp1expected:
            total = total + 1
        else:
            continue

        fwl = 4
        swl = 2

        if len(slp1word1) > fwl:
            excess = len(slp1word1) - fwl
            slp1word1 = slp1word1[-fwl:]
            slp1expected = slp1expected[excess:]
        if len(slp1word2) > swl:
            excess = len(slp1word2) - swl
            slp1word2 = slp1word2[0:swl]
            slp1expected = slp1expected[:-excess]

        if(maxlen < len(slp1expected)):
            maxlen = len(slp1expected)

        difflen = len(slp1expected) - (len(slp1word1) + len(slp1word2))

        if difflen < 2 and difflen > -3:
            word1list.append(slp1word1)
            word2list.append(slp1word2)
            outputlist.append(slp1expected)

        #print(slp1expected + ' => ' + slp1word1 + ' + ' + slp1word2)
        #print(expected + ' => ' + word1 + ' + ' + word2)

    #print(total)
    #print(maxlen)
    return word1list, word2list, outputlist

def get_xy_data(datafile):
    w1l, w2l, ol = get_sandhi_dataset(datafile)
    return w1l, w2l, ol
