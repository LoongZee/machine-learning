# encoding = utf-8

import numpy as np
import pandas as pd
from collections import Counter

#load phase vector 50dim
def loadGloVe(filename):
    vocab = []
    embd = []
    vocab.append('unknownword')
    embd.append([0.0]*50)
    file = open(filename,'r', encoding='UTF-8')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append([float(item) for item in row[1:]])
    file.close()
    return dict(zip(vocab,embd))

def get_sentimentdata():

    # read stopword
    stopwordPath = 'Dataset/stopword.txt'
    stopwordPd = pd.read_table(stopwordPath, error_bad_lines=False, sep=',',names=['stopword'])
    stopword = set(list(stopwordPd['stopword']))
    # read raw data
    sentencePath = 'Dataset/datasetSentences.txt'
    sentencePd = pd.read_table(sentencePath, error_bad_lines=False, sep='\t', header=0 )
    sentencePd['sentence'] = [ i.lower() for i in sentencePd['sentence']]
    dictPath = 'Dataset/dictionary.txt'
    dictPd = pd.read_table(dictPath, error_bad_lines=False, sep='|', names=['sentence','id'] )
    dictPd['sentence'] = [ i.lower() for i in dictPd['sentence']]
    sentimentPath = 'Dataset/sentiment_labels.txt'
    sentimentPd = pd.read_table(sentimentPath, error_bad_lines=False, sep='|', header=0 )

    # merge data
    mergePd_1 = pd.merge(sentencePd, dictPd, how ='inner', left_on=['sentence'], right_on=['sentence'])
    mergeData = pd.merge(mergePd_1, sentimentPd, how ='inner', left_on=['id'], right_on=['phrase ids'])
    allSentence_value = mergeData[['sentence','sentiment values']]
    print ('allSentence_value length: %d' % len(allSentence_value['sentence']))

    # split sentence
    allPhase = []
    for sentence in allSentence_value['sentence']:
        spiltData = sentence.split(' ')
        allPhase.extend(spiltData)
    print ('allPhase length: %d' % len(allPhase))

    # Statist word frequency
    counterPhase = Counter(allPhase)
    uniquePhase = []
    for item in counterPhase.most_common():
        uniquePhase.append(item[0])
    print ('uniquePhase length: %d' % len(uniquePhase))

    #load trained vector
    filename = 'Dataset/glove.6B/glove.6B.50d.txt'
    dictVocab = loadGloVe(filename)
    uqPhase = []
    uqPhasevec = []
    num = 0
    fre = 0
    for i in uniquePhase:
        if i in dictVocab:
            uqPhase.append(i)
            uqPhasevec.append(dictVocab[i])
        else:
            num = num + 1
            fre = fre + counterPhase[i]
            # uniquePhasevec.append(dictVocab['unknownword'])
    dictPhase = dict(zip(uqPhase,uqPhasevec))
    print ('can not find word: %d, frequent: %d' % (num, fre))

    # add vector length label
    Sentence_vector = []
    Sentence_vectorLen = []
    label = []
    maxLen = -1
    for i in range(len(allSentence_value['sentence'])):
        spiltData = allSentence_value['sentence'][i].split(' ')
        val = allSentence_value['sentiment values'][i]
        vec = []
        for j in spiltData:
            if j in dictPhase and j not in stopword:
                vec.append(dictPhase[j])
        if len(vec) > 0:
            Sentence_vector.append(vec)
            Sentence_vectorLen.append(len(vec))
            if val < 0.2:
                label.append(0)
            elif val < 0.4:
                label.append(1)
            elif val < 0.6:
                label.append(2)
            elif val < 0.8:
                label.append(3)
            else:
                label.append(4)
            if maxLen < len(vec):
                maxLen = len(vec)
    for i in range(len(Sentence_vector)):
        if len(Sentence_vector[i]) < maxLen:
            padding = [[0.0]*50]*(maxLen - len(Sentence_vector[i]))
            Sentence_vector[i].extend(padding)

    # allSentence_value.insert(2,'vec',Sentence_vector)
    # allSentence_value.insert(3,'length',Sentence_vectorLen)
    # allSentence_value.insert(4,'label',label)
    return np.array(Sentence_vector), np.array(Sentence_vectorLen), np.array(label)