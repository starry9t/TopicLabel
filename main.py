#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:11:33 2019

@author: Yu Zhou
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:02:41 2019

@author: Yu Zhou
"""


# In[1]:
import os,sys
cur_dir = os.getcwd()
sys.path.append('SIF/')
import data_io, params, SIF_embedding
import pickle
import csv, time, re
import nltk
#nltk.download('wordnet')
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize as wt
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
import enchant
endict = enchant.Dict("en_UK")
import numpy as np
import pandas as pd
import spacy
import heapq
import operator
from collections import defaultdict
import gensim 
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from fastText import train_unsupervised
from fastText import load_model
# Here we add Gensim libraries 
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
###Onlyl First Time   ####  python3 -m spacy download en    #### https://spacy.io/api/annotation
nlp = spacy.load('en', disable=['parser', 'ner']) #Initialize spacy 'en' model, keeping only tagger component (for efficiency)
#Prepare Stopwords: Download and import them and make it available in stop_words.


# In[2]: SET ALL VARIABLES ACCORDING TO YOUR REQUIREMENT

# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
# Update this path to where you saved mallet
#mallet_path = '/home/user/anaconda3/mallet-2.0.8/bin/mallet' # this path is runned on local machine
mallet_path = '/home/zhouyu/anaconda3/mallet-2.0.8/bin/mallet' # this path is runned on server

# number of topics, estimate this number in [startN,  limitN)
startN = 20
limitN = 78
stepN = 5
SMOOTH = 0.1 # # to avoid the consequence of vector norm equals to 0
D = 50 # dimentional number of vector
EXP = 'B'  # in ['A', 'B', 'c'] see line 133
modeltag = 'expA'  # tag it for different original files or Lda models

# In[3]: SET ALL INTERMEDIATE FOLDER AND FILENAMES
#### you don't have to edit this part.
#-- folders 
#
inter_dir = os.path.join(os.getcwd(),'Interfile')
if os.path.isdir(inter_dir):
    pass
else:
    os.mkdir(inter_dir)
#    
model_dir = os.path.join(cur_dir , 'MalletModels')
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
#    
outputDir = os.path.join(cur_dir,'Output')
if os.path.isdir(outputDir):
    pass
else:
    os.mkdir(outputDir)
#
data_dir = os.path.join(cur_dir,'data')
#-- filenames
original_file = os.path.join(data_dir,'clinical.txt')
coherence_file = os.path.join(model_dir,'CV_Coherence_list.txt')

#glove300_txt_file = os.path.join(data_dir, 'glove.840B.300d.txt')
#gloveWeight_file = os.path.join(data_dir, 'enwiki_vocab_min200.txt')
word2weight_pickle_file = os.path.join(data_dir,'word2weight.pickle')

embedding_txt_file = os.path.join(inter_dir, 'embedding.txt')
processed_pickle_file = os.path.join(inter_dir, 'processed.pickle')
processed_txt_file = os.path.join(inter_dir, 'processed.txt')
id2word_file = os.path.join(inter_dir,'id2word.pickle')
tdmatrix_file = os.path.join(inter_dir,"topic_doc_matrix.npy")
twmatrix_file = os.path.join(inter_dir,"topic_word_matrix.npy")
docN = 20
topNdoc_file = os.path.join(inter_dir, 'top{}_doc_idx.txt'.format(docN))
wordN = 10
topNwordprob_file = os.path.join(inter_dir, 'top{}_word_prob.pickle'.format(wordN))
ftmodel_file = os.path.join(inter_dir,'ftmodel.bin') # fastText model
label_file_word = os.path.join(outputDir,'topicLabels_Words.txt')

#phrase_list_file = os.path.join(inter_dir, 'phrasesList.pickle')
phrase_dict_file = os.path.join(inter_dir, 'phrasesDict.pickle')
#phrase_txt_file = os.path.join(inter_dir, 'phrases.txt')
new_txt_file = os.path.join(inter_dir, 'new.txt')
#data_word_clean_txt_file = os.path.join(inter_dir, 'dwctxt.txt')
#glove300_dict_file = os.path.join(inter_dir, 'gloveDict.pickle')
#We_npy_file = os.path.join(inter_dir, 'We.npy')
#words_pickle_file =os.path.join(inter_dir, 'words.npy')
corpusWeight_file = os.path.join(inter_dir, 'clinic_vocab_min3.txt')

# In[4]:

### This part contains all the functions that we need. 

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def remove_nonwords(datawords):
    try:
        for s in datawords:
            try:
                i = 0
                while s[i]:
                    if not endict.check(s[i]):
                        del s[i]
                    else:
                        i += 1
            except:
                pass
    except:
        pass
    return datawords

def remove_stopwords(texts):
    return [[word for word in doc if word not in stop_words] for doc in texts]

def remove_pun(data):
    new_data = []
    for line in data:
        new_line = []
        for item in line:
            if item[0].isalpha():
                new_line.append(item)
            else:
                pass
        new_data.append(new_line)
    return new_data

def writeNewTxt(new_data, new_file):
    with open(new_file, 'w') as etf:
        for line in new_data:
            for word in line:
                etf.write('{} '.format(word))
            etf.write('\n')
    etf.close()
    print('processed txt written.')    
    return

def preprocessTxt(txtfile, newfile):
    with open(txtfile, 'r') as reader:
        data = reader.readlines()
    reader.close()
    new_data = []
    renew_data = []
    pro_data = []
    for line in data:
        sents = wt(line.strip())
        new_data.append(sents)
    with open(newfile, 'w') as etf:
        for line in new_data:
            renew_line = []
            pro_line = []
            for word in line:
                word = str.lower(word)
                renew_line.append(word)
                etf.write('{} '.format(word))
                if word[0].isalpha():
                    pro_line.append(word)
                else:
                    pass
            renew_data.append(renew_line)
            pro_data.append(pro_line)
            etf.write('\n')
    etf.close()
    print('txtfile preprocessed.')       
    
    return renew_data, pro_data

def buildPhrases(processed_data):
    phraseDict = defaultdict(int)
    for doc in processed_data:
        for item in doc:         
            if '_' in item:
                phraseDict[item] += 1
    with open(phrase_dict_file, 'wb') as phdf:
        pickle.dump(phraseDict, phdf)
    phdf.close()
    
    print('there are {} phrases in phrase dict.'.format(len(phraseDict)))
    return

def buildCorpusFreq(txtfile, corpusWeight_file):
    with open(txtfile, 'r') as reader:
        lines = reader.readlines()
    reader.close()
    wordDict = defaultdict(int)
    for line in lines:
        line = wt(line, language='english')
        for word in line:
            wordDict[word] += 1
        #sents = sent_tokenize(line, language='english')
#        for sent in sents:
#            for word in sent:
#                wordDict[word] += 1
    # order dict to list
    wordFreq = sorted(wordDict.items(), key=operator.itemgetter(1), reverse=True)
    
    with open(corpusWeight_file, 'w') as writer:
        for (w, f) in wordFreq:
            if int(f) < 3:
                break
            writer.write('{} {}\n'.format(w, f))
    writer.close()
    return

def buildW2VDict(method='FastText'):
    w2vdict = {} # the value should be (D, ) ndarray
    
    # there should be a pretrained load way  
    if method == 'FastText': #https://fasttext.cc/docs/en/english-vectors.html  download wordvectors
        if os.path.isdir(ftmodel_file):
            w2vmodel = load_model(ftmodel_file)
        else:
            w2vmodel = train_unsupervised(embedding_txt_file, model='skipgram', lr=0.05, dim=D, ws=2, epoch=5, minCount=2, 
                                          minCountLabel=0, minn=3, maxn=6, neg=5, wordNgrams=3, loss='ns', bucket=2000000, 
                                          thread=5, lrUpdateRate=100, t=0.0001, label='__label__', verbose=2, pretrainedVectors='')
            w2vmodel.save_model(ftmodel_file)
        print('fastText word vector model trained as ftmodel.bin')
        # Turn it into dict
        (word,freq) = w2vmodel.get_words(include_freq=True)
        for w, f in zip(word, freq):
            w2vdict[w] = w2vmodel.get_word_vector(w)
    print('w2vdict built.')
    return w2vdict

def fromTxt2Docs(txtfile):   #
    '''
    This function will do:
        1 process the original txtfile
        2 return a processed data(list of list) to form the input of LdaMallet model
        3 build phrase dictionary and write it into txt
        4 write a processed txt file to count word frequency
        5 write the word frequency txt as the input of SIF
    '''
    if os.path.isfile(processed_pickle_file):
        with open(processed_pickle_file, 'rb') as f:
            processed_data = pickle.load(f)
        f.close()
        print('processed_data loaded.')
        return processed_data

    # Preprocess txt
    new_data, pro_data = preprocessTxt(txtfile, new_txt_file)
    
    pro_data = remove_nonwords(pro_data)
    pro_data = remove_stopwords(pro_data)  # Remove Stop Words
    print("data words preparing...")
    bigram = gensim.models.Phrases(pro_data, min_count=30, threshold=0.1, delimiter=b'_')# higher threshold fewer phrases.
    print("get bigram")
    trigram = gensim.models.Phrases(bigram[pro_data], min_count=30, threshold=0.1, delimiter=b'_')#, threshold=100)  
    print("get trigram")    
    
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    print("get bigram_mod")
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    print("get trigram_mod")

    data_words_trigrams = [trigram_mod[bigram_mod[doc]] for doc in new_data]  # Form Trigrams
    
    # Write a new txt file which contains phrases 
    writeNewTxt(data_words_trigrams, embedding_txt_file)  # build w2v dict
    
    # write word/phrase frequency
    buildCorpusFreq(embedding_txt_file, corpusWeight_file)

    # build phrase list and dict, write dict txt.
    buildPhrases(data_words_trigrams)
    
    processed_data = remove_stopwords(data_words_trigrams)  
    processed_data = remove_pun(processed_data)
    processed_data = lemmatization(processed_data)
    
    with open(processed_pickle_file, 'wb') as writer:
        pickle.dump(processed_data, writer)
    writer.close()    
    print("----data processed----")
    print('processed data[0]:\n{}'.format(processed_data[0]))
    
    writeNewTxt(processed_data, 'test.txt')

    return processed_data
    
def FormInput4Mallet(processed_data):
    
    if os.path.isfile(id2word_file):
        with open(id2word_file,'rb') as i2w:
            id2word = pickle.load(i2w)
        print('id2word file is loaded.')  
        i2w.close()    
        corpus = [id2word.doc2bow(text) for text in processed_data]
        return id2word, corpus
    
    #---  
    print('Start forming input files for LdaMallet model...')
    id2word = corpora.Dictionary(processed_data)
    with open(id2word_file,'wb') as i2w:
        pickle.dump(id2word,i2w)
    print("get id2word and save it to \'id2word\' file. id2word dictionary has {} words.".format(len(id2word)))
    i2w.close()
    
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in processed_data]
    print("get corpus")
    
    return id2word, corpus

def runMalletModels(dictionary, corpus, texts, limit=20, start=2, step=3):
    
    model_list = []
    coherence_values = []
    j = 0
    for i in range(60):   
        file = modeltag + str(i) + ".mallet"
        filename =  os.path.join(model_dir,file)
        if os.path.isfile(filename):
            model_list.append(gensim.models.wrappers.LdaMallet.load(filename,mmap='r'))
            j += 1
    if len(model_list) > 1: # means model list loaded.
        coherence_values = []
        with open(coherence_file, "r") as f:
            for line in f:
                coherence_values.append(float(line.strip()))
        f.close()
        print("------ {} mallet models and CV coherences of them are loaded!".format(j))     
        return model_list, coherence_values

    #    
    for num_topics in range(start, limit, step):
        start_time = time.time()
        print("Training LDA with {0} topics starts ...".format(num_topics) )
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_ldamallet = coherence_model.get_coherence()
        coherence_values.append(coherence_ldamallet)
        print("LDA training time = %s min" % ((time.time() - start_time)/60))
        print('Coherence Score: ', coherence_ldamallet)
    print("there are {0} models learned.".format(len(model_list)))

    for i,model in enumerate(model_list):
        fn = modeltag + str(i) + ".mallet"
        filename =  os.path.join(model_dir, fn)
        model.save(filename)
    print("models saved.")
    
    with open(coherence_file,"wt") as f:
        for C in coherence_values:
            f.write(str(C) +"\n")
    print("CV coherences of models are saved.")
    f.close()
    
    return model_list, coherence_values


def generateInterFiles(optimal_model,corpus):
    num_topics = optimal_model.num_topics # Number of topics in optimal model
    #
    print("Starting constructing the Document-Topic probability matrix...")
    topic_doc_matrix = np.zeros((num_topics, len(corpus))) 
    ######
    
    
    for i, row in enumerate(optimal_model[corpus]):
        ##################

        ###################
        row = sorted(row, key=lambda x: (x[0]))
        for topic,prop in row:
            topic_doc_matrix[topic][i] = prop   
    np.save(tdmatrix_file, topic_doc_matrix, allow_pickle=False) 
    print("a {} Document-Topic probability matrix is constructed and saved.".format(np.shape(topic_doc_matrix)))
    #
    print("Start constructing the Word-Topic probability matrix...")
    topic_word_matrix = optimal_model.get_topics()
    np.save(twmatrix_file, topic_word_matrix, allow_pickle=False)
    print("a {} Word-Topic matrix is constructed and saved.".format(np.shape(topic_word_matrix)))
    #
    # Save the Id of the top 10 relevant document for each topic 
    topN_doc = []
    for x in topic_doc_matrix:
        a = [x for x in (-x).argsort()[:docN]]
        topN_doc.append(a)
    np.savetxt(topNdoc_file, topN_doc, fmt='%s')
    print("top{}_doc.txt is written.".format(docN))

    # Save the top N(wordN) relevant words and their ids and probabilities for each topic
    topN_word_prob = []
    topN_word = []
    ti = 0
    for x in topic_word_matrix:
        b = [x for x in (-x).argsort()[:wordN]]
        totalprob = 0
        wlist = []
        for wordidx in b:
            totalprob += topic_word_matrix[ti][wordidx]
            word = id2word[wordidx]
            wlist.append(word)
        topN_word.append(wlist)
        c = []
        for wordidx in b:
            (wordid, prob) = (wordidx, topic_word_matrix[ti][wordidx]/totalprob)
            c.append((wordid,prob))
        topN_word_prob.append(c)
        ti += 1
    # Above we get topN_word_prob and topN_word
    
    # topN_word_prob pickle
    with open(topNwordprob_file,'wb') as ts:
        pickle.dump(topN_word_prob,ts)
    ts.close()
    
    # 
    with open(label_file_word, 'w') as tnw:
        for i in range(num_topics):
            tnw.write('Topic {}:\n'.format(i))
            for word in topN_word[i]:
                tnw.write('{}\t'.format(word))
            tnw.write('\n------\n')
        print('Topic labeling with words is written in {}.'.format(label_file_word))
    tnw.close()

    #this part can be used for manual check. you can comment it.        
    t1wpf = os.path.join(inter_dir, 'top{}wordprob.txt'.format(wordN))
    with open(t1wpf, 'w') as ww:
        for row in topN_word_prob:
            ww.write(str(row))
            ww.write('\n')
    ww.close()
    print('top10_word_prob.txt is written.')
    return num_topics

def getWordVector(word):
    try:
        v = Di[word]
        return v
    except:
        v = np.ndarray((D,))
        return v

def getTopicVector(topicid):
    topicVector = np.zeros(D)
    
    with open(topNwordprob_file,'rb') as twf:
        topicwords = pickle.load(twf)
   
    wnp = topicwords[topicid]
    for (wordid, prob) in wnp:
        word = id2word[int(wordid)]
        wordVector = getWordVector(word)
        wordVector = wordVector*prob
        topicVector += wordVector
    return topicVector

def getTopicVectors(n):
    tvdict = {}
    for i in range(n):
        tv = getTopicVector(i)
        tvdict[i] = tv
    return tvdict  

def getSentenceVector(sentence):
    sentenceVector = np.zeros(D)
    wc = 0
    for word in sentence:
        wc += 1
        wordVector = getWordVector(word)
        sentenceVector += wordVector
    sentenceVector = sentenceVector/wc
    return sentenceVector

def getSIFSentenceVector(sifEmbed, docNum, sentNum):
    
    return 

def similarityTopicSentence(tvector,svector):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a,b in zip(tvector,svector):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0:
        return SMOOTH
    elif (normB == 0.0):
        return SMOOTH
    else:
        return dot_product/((normA*normB)**0.5)
    
def buildTWDScoreDict(topic_word_matrix, id2word):
    twdsdict = defaultdict(float)
    (topicnumber,wordnumber) = np.shape(topic_word_matrix)
    for topicid in range(topicnumber):
        for wordid in range(wordnumber):
            word = id2word[wordid]
            t_prime =[topic_word_matrix[z][wordid] for z in range(topicnumber) if z != topicid]
            fenmu = max(t_prime)
            twdsdict[(topicid,word)] = topic_word_matrix[topicid][wordid]/fenmu
    print('a topic word dscore dictionary is built.')
    return twdsdict

def sentDiscriminativeScore(sentence, topicid, twdsdict):
    sentence_score = 0
    for word in sentence:
        sentence_score += twdsdict[(topicid,word)]
    return sentence_score/len(sentence)

def generateLabelFile(txtfile, Nsent=5, strategy = 'DiscriminativeScore', twdsdict={}):
    '''
    -----strategy: Similarity
    assume one document is focus on only 1 topic.
    then calculate similarity between this topic and sentences in documents focused on this topic.
    then pick top S sentences
    -----strategy: DiscriminativeScore
    xxxxx
    '''
    label_file_sent = os.path.join(outputDir,'topicLabels_Sentences_{}.txt'.format(strategy))
    
    with open(txtfile,'r',encoding='utf-8') as ad:
        alldocs = ad.read().splitlines()
    print('txt file is loaded in list.')
    ad.close()
    tsDict = {}
    topicdocs = np.genfromtxt(topNdoc_file)
    N = len(topicdocs)
    topicVectorDict = getTopicVectors(N)
    
    if strategy == 'SIFSimilarity':
        for topicid in range(N):
            labelList = []
            tv = topicVectorDict[topicid]
            docidlist = topicdocs[topicid]
            for docid in docidlist:
                docid = int(docid)
                doc = alldocs[docid]
                doc = doc.strip()
                sents = sent_tokenize(doc)
                for (i, sent) in enumerate(sents):
                    sv = DocSentVectorD[docid][i]
                    similarity = similarityTopicSentence(tv, sv)
                    labelList.append((similarity, sent))
                labelList = heapq.nlargest(Nsent, labelList, key=lambda s:s[0])
            labelList = heapq.nlargest(Nsent, labelList, key=lambda s:s[0])
            tsDict[topicid] = labelList

    if strategy == 'Similarity':   
        #pick the topic
        for topicid in range(N):
            labelList = []
            tv = topicVectorDict[topicid]
            #print(tv)
            docidlist = topicdocs[topicid]
            for docid in docidlist:
                docid = int(docid)
                doc = alldocs[docid]  #get text
                sents = sent_tokenize(doc, language='english')
                for sent in sents:
                    sv = getSentenceVector(sent)
                    similarity = similarityTopicSentence(tv,sv)
                    labelList.append((similarity,sent))
                labelList = heapq.nlargest(Nsent,labelList, key=lambda s:s[0])
            labelList = heapq.nlargest(Nsent,labelList, key=lambda s:s[0])
            tsDict[topicid] = labelList   

    if strategy == 'DiscriminativeScore':
        #pick the topic
        for topicid in range(N):
            labelList = []
            tv = topicVectorDict[topicid]
            docidlist = topicdocs[topicid]
            for docid in docidlist:
                docid = int(docid)
                doc = alldocs[docid]
                sents = sent_tokenize(doc, language='english')
                c = min(len(sents),Nsent)
                for sent in sents:
                    dscore = sentDiscriminativeScore(sent, topicid, twdsdict)
                    labelList.append((dscore,sent))
                labelList = heapq.nlargest(c,labelList, key=lambda s:s[0])
            labelList = heapq.nlargest(Nsent,labelList, key=lambda s:s[0])
            tsDict[topicid] = labelList

            
    ###
    with open(label_file_sent, 'w') as tl: 
        for i in range(len(tsDict)):  
            tl.write('topic {}:\n'.format(i))
            for j in range(len(tsDict[i])):
                tl.write('{}\n'.format(tsDict[i][j][1]))
            tl.write('------\n')
    print('topic labeling with sentences is written in {}.'.format(label_file_sent))
    tl.close()
    return

def getWordMap(w2vDic):
    n = 0
    words = {}
    We = []
    for (w, v) in w2vDic.items():
        words[w] = n
        n += 1
        We.append(v)
    We = np.array(We)

    print('In getWordmap function, we get two returns:')
    print('words: length is {}'.format(len(words)))
    print('We: shape is {}'.format(We.shape))
    print('dtype of we is: {}'.format(We.dtype))
    return (words, We)

def SIFSentEmbedding(weighttxt, docfile, words, We, weight4ind,
                     weightpara = 1e-3, paramm = 1):
     # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
     # number of principal components to remove in SIF weighting scheme
    sentences = sent_tokenize(docfile)
    x, m = data_io.sentences2idx(sentences, words)
    w = data_io.seq2weight(x, m, weight4ind) # get word weights
    paramm = params.params()
    paramm = paramm.LC
    embedding = SIF_embedding.SIF_embedding(We, x, w, paramm) # embedding[i,:] is the embedding for sentence i
    return embedding


def SIFDocEmbedding(w2vdict, weighttxt, txtfile):
    
    weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    (words, We) = getWordMap(w2vdict)
    word2weight = data_io.getWordWeight(weighttxt, word2weight_pickle_file, weightpara) # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word    
    
    DocVectorDict = {}
    DocSentVecDict = {}
    docNum = 0
    with open(txtfile, 'r') as reader:
        txt = reader.readlines()
    reader.close()
    for doc in txt:
        doc = doc.strip()
        sentEm = SIFSentEmbedding(weighttxt, doc, words, We, weight4ind,
                     weightpara = 1e-3, paramm = 1)
        DocSentVecDict[docNum] = sentEm
        docVector = (np.sum(sentEm, axis=1))/(sentEm.shape[0]) 
        DocVectorDict[docNum] = docVector
        docNum += 1
    return DocVectorDict, DocSentVecDict, We



def GeneratePhraseLabel(w2vmodel, topicNum):
    
#==== when you have vectors 
    #download phrase dict
    with open(phrase_dict_file, 'rb') as phdf:
        phraseDict = pickle.load(phdf)
    phdf.close()
    #build phrase vector dict
    phraseVectors = {}
    for item in phraseDict:
        v = w2vmodel.get_word_vector(item)
        phraseVectors[item] = v
    #get N topics and their vectors.
    topicVectors = getTopicVectors(topicNum)
    #compute similarity between phrases and topic
    phraseLabelDict = {}
    for i in range(topicNum):
        topicVector = topicVectors[i]
        phraseLabels = []
        for phrase in phraseVectors:
            sml = similarityTopicSentence(topicVector,phraseVectors[phrase])
            phraseLabels.append((sml,phrase))
        top10phrases = heapq.nlargest(10,phraseLabels, key=lambda s:s[0])
        phraseLabelDict[i] = top10phrases
    
    ###
    with open('phraseLabel.txt', 'w') as tl: 
        for i in range(len(phraseLabelDict)):
            tl.write('topic {}:\n'.format(i))
            for j in range(len(phraseLabelDict[i])):
                tl.write('{}\t'.format(phraseLabelDict[i][j][1]))
            tl.write('\n------\n')
    print('topic labeling with phrases is written in phraseLabel.txt.')
    tl.close()
    
    return

# In[303]:

if __name__ == '__main__':


    # you can run this step by step or all in once.
    processed_data = fromTxt2Docs(original_file)
    print('Now we got processed data.')
    (id2word, corpus) = FormInput4Mallet(processed_data) 
    (model_list, coherence_values) = runMalletModels(dictionary=id2word, corpus=corpus, texts=processed_data, start=startN, limit=limitN, step=stepN)
    optimal_model = model_list[coherence_values.index(max(coherence_values))]  
    topicNum = generateInterFiles(optimal_model,corpus)  # This will also generate one of the output file --> topicLabels_Words.txt
    print('Now we get all intermediate files.')
#    
    Di = buildW2VDict('FastText') 
    w2vmodel = load_model(ftmodel_file)
    GeneratePhraseLabel(w2vmodel, 20)
    
    generateLabelFile(txtfile=embedding_txt_file, Nsent=5, strategy = 'Similarity')
    
    twmatrix = np.load(twmatrix_file)
    twdsdict = buildTWDScoreDict(twmatrix,id2word)
    generateLabelFile(txtfile=embedding_txt_file, Nsent=5, strategy = 'DiscriminativeScore', twdsdict=twdsdict)


    DocVectorD, DocSentVectorD, We = SIFDocEmbedding(Di, corpusWeight_file, embedding_txt_file)
    generateLabelFile(txtfile=embedding_txt_file, Nsent=5, strategy = 'SIFSimilarity', twdsdict=twdsdict)
    
    GeneratePhraseLabel(w2vmodel, 20)
    
    print('There you go!')

    
    