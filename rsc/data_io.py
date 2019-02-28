from __future__ import print_function

import numpy as np
import pickle
import os
import tree

#from theano import config

#def getWordmap(textfile):
#    words={}
#    We = []
#    f = open(textfile,'r')
#    lines = f.readlines()
#    for (n,i) in enumerate(lines):
#        i=i.strip().split(' ')
#        if len(i) == 301:
#            v = []
#            for num in i[1:]:
#                v.append(float(num))
#            words[i[0]]=v
#        We.append(v)
#        return (words, np.array(We))

def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask

def lookupIDX(words,w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in words:
        return words[w]
    elif 'UUUNKKK' in words:
        return words['UUUNKKK']
    else:
        return len(words) - 1

def getSeq(p1,words):
    p1 = p1.split()
    X1 = []
    for i in p1:
        X1.append(lookupIDX(words,i))
    return X1

def sentences2idx(sentences, words):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location)
    """
    seq1 = []
    for sent in sentences:
        seq1.append(getSeq(sent,words))  # seq is a list of word indices which are in sentences
    x1,m1 = prepare_data(seq1)
#    print('x shape: {}\nm shape: {}'.format(x1.shape, m1.shape))
    return x1, m1

def getWordWeight(weightfile, word2weight_pickle_file, a=1e-3):
#    if os.path.isfile(word2weight_pickle_file):
#        with open(word2weight_pickle_file, 'rb') as reader:
#            word2weight = pickle.load(reader)
#        reader.close()
    if a==1:
        print('hah')
    else:
        if a <=0: # when the parameter makes no sense, use unweighted
            a = 1.0
    
        word2weight = {}
        with open(weightfile) as f:
            lines = f.readlines()
        N = 0
        for i in lines:
            i=i.strip()
            if (len(i)>0):
                i = i.split()
                if(len(i) == 2):
                    word2weight[i[0]] = float(i[1])
                    N += float(i[1])
                else:
                    #print(i)
                    pass
        for key, value in word2weight.items():
            word2weight[key] = a / (a + value/N)
        with open(word2weight_pickle_file, 'wb') as writer:
            pickle.dump(word2weight, writer)
        writer.close()
    print('getWordWeight. length of word2weight is {}'.format(len(word2weight)))
    return word2weight

def getWeight(words, word2weight):
    weight4ind = {}
    for word, ind in words.items():
        try:
            weight4ind[ind] = word2weight[word]
        except:
            #print('word: {}   ind: {}'.format(word,ind))
            weight4ind[ind] = 1.0
    print('getWeight executed. Length: {}'.format(len(weight4ind)))
    for item in weight4ind:
        if weight4ind[item] != 1:
            print('Not all 1.')
            break
    return weight4ind

def seq2weight(seq, mask, weight4ind):
    weight = np.zeros(seq.shape).astype('float32')
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i,j] > 0 and seq[i,j] >= 0:
                weight[i,j] = weight4ind[seq[i,j]]
    weight = np.asarray(weight, dtype='float32')
    return weight
