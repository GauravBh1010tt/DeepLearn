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

maxcompoundlen = 50
inwordlen = 5

def remove_nonslp1_chars(word):
    newword = ''
    for char in word:
        if char in slp1charlist:
            newword = newword + char
    return newword

def get_sandhi_dataset(datafile):
    datalist = []

    with open(datafile) as fp:
        tests = fp.read().splitlines()
    
    total = 0
    maxlen = 0
    minlen = 500
    count = [0, 0, 0, 0, 0, 0]

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

        fwl = 2
        swl = 2

        start = 0
        end = len(slp1expected)

        fullslp1expected = slp1expected
        fullslp1word1 = slp1word1
        fullslp1word2 = slp1word2

        if len(slp1word1) > fwl:
            start = len(slp1word1) - fwl
        if len(slp1word2) > swl:
            end = end - len(slp1word2) + swl

        difflen = len(slp1expected) - (len(slp1word1) + len(slp1word2))

        if difflen < 2 and difflen > -3 and len(slp1expected) > len(slp1word1) and len(slp1expected) > len(slp1word2) and len(fullslp1expected) <= maxcompoundlen and len(fullslp1expected) >= inwordlen:
            total = total + 1

            startblock = False
            endblock = False

            while end-start < inwordlen:
                if start > 0:
                    start = start - 1
                else:
                    startblock = True
                if end-start == inwordlen:
                    break
                if end < len(slp1expected):
                    end = end + 1
                else:
                    endblock = True
                if end-start == inwordlen:
                    break
                if startblock and endblock:
                    break

            newlen = len(slp1word2) - len(slp1expected) + end

            if slp1word1[:start] == slp1expected[:start] and slp1word2[newlen:] == slp1expected[end:]:
                slp1word1 = slp1word1[start:]
                slp1word2 = slp1word2[:newlen]
                slp1expected = slp1expected[start:end]

                datalist.append([slp1word1, slp1word2, slp1expected, fullslp1expected, start, end, fullslp1word1, fullslp1word2])

                if(maxlen < end-start):
                    maxlen = end-start
                if(minlen > end-start):
                    minlen = end-start

                count[end-start] = count[end-start] + 1
            """
            else:
                print(slp1word1)
                print(slp1word2)
                print(slp1expected)
                print("*****************************")
            """
        #print(slp1expected + ' => ' + slp1word1 + ' + ' + slp1word2)
        #print(expected + ' => ' + word1 + ' + ' + word2)

    print(total)
    print(maxlen)
    print(minlen)
    print(count)
    return datalist

def get_xy_data(datafile):
    dl = get_sandhi_dataset(datafile)
    return dl
