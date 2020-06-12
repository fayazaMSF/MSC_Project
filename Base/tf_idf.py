import math
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity



def get_doc(sent):
    doc_info = []
    for sent in sent:
        # print (sent)
        i = sent.split("\t")[0]
        snet1 = sent.split("\t")[1]
        count = count_words(snet1)
        temp = {'doc_id': i, 'doc_length': count}
        doc_info.append(temp)
    return doc_info


def count_words(sent):
    count = 0
    words = sent.split()
    # print(sent)
    # word_tokenize(sent)
    for word in words:
        count += 1
    return count


def create_freq_dict(sents):
    # i = 0
    freqDict_List = []
    for sent in sents:
        i = int(sent.split("\t")[0])
        freq_dict = {}
        words = sent.split("\t")[1].split()
        # print(words)
        for word in words:
            # word = word.lower()
            if word in freq_dict:
                freq_dict[word] += 1
            else:
                freq_dict[word] = 1
            temp = {'doc_id': i, 'freq_dict': freq_dict}
            # print(temp)
        freqDict_List.append(temp)
    # print(freqDict_List)
    return freqDict_List

# tf(t,d) = count of t in d / number of words in d
def computeTF(doc_info, freqDict_List_):
    TF_scores = []
    for tempDict in freqDict_List_:
        # print(tempDict)
        id_ = tempDict['doc_id']
        # print(id_)
        for k in tempDict['freq_dict']:
            # print(k)
            # print(tempDict['freq_dict'][k])
            # print(doc_info[id_ -1]['doc_length'])
            temp = {'doc_id': id_,
                    'TF_score': tempDict['freq_dict'][k] / doc_info[id_ - 1]['doc_length'],
                    'key': k}
            # print(temp)
            TF_scores.append(temp)
    return TF_scores


def computeIDF(doc_info, freqDict_List):
    IDF_scores = []
    counter = 0
    for dict in freqDict_List:
        counter += 1
        for k in dict['freq_dict'].keys():
            count = sum([k in tempDict['freq_dict'] for tempDict in freqDict_List])
            temp = {'doc_id': counter, 'IDF_score': math.log(len(doc_info) / count), 'key': k}
            # print(temp)
            IDF_scores.append(temp)
    return IDF_scores


def computeTFIDF(TF_scores, IDF_scores):
    TFIDF_scores = []
    for j in IDF_scores:
        # print("------------------------")
        # print(j)
        for i in TF_scores:
            # print(i)
            # print(j['doc_id'] == i['doc_id'] and j['key'] == i['key'])
            if j['doc_id'] == i['doc_id'] and j['key'] == i['key']:
                temp = {'doc_id': j['doc_id'],
                        'TFIDF_score': j['IDF_score'] * i['TF_score'],
                        'key': i['key']}

                # print(temp)
        TFIDF_scores.append(temp)
        # print("***********************************"+TFIDF_scores.__sizeof__())
    return TFIDF_scores

def sentece_computeTFIDF(TFIDF_scores,Doc_Info,sents):
    sentTFIDF_scores = []
    # np.array.tv_matrix = []
    for sent in sents:
        i = int(sent.split("\t")[0])
        freq_dict = {}
        words = sent.split("\t")[1].split()
        # print(words)
        count = 0
        sum = 0.00
        for word in words:
            # print(TFIDF_scores[i][word])
            for tfidf_ in TFIDF_scores:

                if i== int(tfidf_['doc_id']) and word == tfidf_['key']:
                    count += 1
                    # print(count)
                    # print(word)
                    # print(tfidf_['TFIDF_score'])
                    sum += float(tfidf_['TFIDF_score'])
        # print(sum)
        # print(count)
        # tv_matrix.(sum)
        temp = {'doc_id': i, 'sentTFIDF_score':sum, 'key': sent}
        # print(temp)
        sentTFIDF_scores.append(temp)
    return sentTFIDF_scores
    # return tv_matrix
            # word = word.low
    # # print(TFIDF_scores)
    # sentTFIDF_scores = []
    # for dict_ in Doc_Info:
    #     count =0
    #     sum =0.00
    #     for tfidf_ in TFIDF_scores:
    #         # print(tfidf_['doc_id'])
    #         # print(dict_['doc_id'])
    #         # print("**********************")
    #         # print(int(dict_['doc_id']) == int(tfidf_['doc_id']))
    #         if int(dict_['doc_id']) == int(tfidf_['doc_id']):
    #             print(tfidf_['TFIDF_score'])
    #             print(tfidf_['key'])
    #             count += 1
    #             print(count)
    #             sum += float(tfidf_['TFIDF_score'])
    #             print(sum)


def cosineCalculation(sentTFIDF_scores):
    sentCos_scores = []
    for sent in sentTFIDF_scores:
        # print(sent)
        for tfIdf in sentTFIDF_scores:
            v1 = sent['sentTFIDF_score']
            v2 = tfIdf['sentTFIDF_score']
            # np.array(sent['sentTFIDF_score'])
        # v2 = np.array(tfIdf['sentTFIDF_score'])
            print(">>>>>>>>>>>>>>>>>>>>>>>." + str(v1) + str(v2))

            sim = np.dot(v1, v2) / (np.sqrt(np.sum(v1*v1)) * np.sqrt(np.sum(v2*v2)))
            print(sim)
            # print(pd.DataFrame(data=sim, index=sent['doc_id'], columns=tfIdf['doc_id']))



            # val = cosine_similarity(v1,v2)
            # print(val)
            # temp = {'doc_id': sent['doc_id'], 'cos_score': sim}
            # sentCos_scores.append(temp)
            # print(temp)

def bagofwords(sentences):
    words = []
    for sentence in sentences:
        word = sentence.split("\t")[1].split()
        words.extend(word)
    output = set()
    for x in words:
        output.add(x)

    words =list(set(words))
    print(words)
    return words

def computeTFIDFVectorForNews(sent):
    tfidfVector = [0.0] * len(bow)
    i = int(sent.split("\t")[0])
    news = sent.split("\t")[1].split()
    # For each unique word, if it is in the review, store its TF-IDF value.
    for i, word in enumerate(bow):
        if word in news:
            tfidfVector[i] = 1
    return tfidfVector


def computeTFIDFVector(bow,TFIDF_scores,sents):
    tfidfVector = [0.0] * len(bow)

    for sent in sents:
        tfidfVector = [computeTFIDFVectorForNews(sent)]
        # for word in words:
        #     # print(TFIDF_scores[i][word])
        #     for tfidf_ in TFIDF_scores:
        #
        #         if i== int(tfidf_['doc_id']) and word == tfidf_['key']:
        #             for j, word in enumerate(bow):
        #                 print(j)
        #                 print("___________"+word)
        #                 tfidfVector[j] =tfidf_['TFIDF_score']
    return tfidfVector

    # # For each unique word, if it is in the review, store its TF-IDF value.
    # for i, word in enumerate(bow):
    #     print("______"+word)
    #     print(i)
    #     if word in review:
    #         tfidfVector[i] = review[word]
    # return tfidfVector


def sentece_computeTFIDF(TFIDF_scores,Doc_Info,sents):
    sentTFIDF_scores = []
    # np.array.tv_matrix = []
    for sent in sents:
        i = int(sent.split("\t")[0])
        freq_dict = {}
        words = sent.split("\t")[1].split()
        # print(words)
        count = 0
        sum = 0.00
        for word in words:
            # print(TFIDF_scores[i][word])
            for tfidf_ in TFIDF_scores:

                if i== int(tfidf_['doc_id']) and word == tfidf_['key']:
                    count += 1
                    # print(count)
                    # print(word)
                    # print(tfidf_['TFIDF_score'])
                    sum += float(tfidf_['TFIDF_score'])
        # print(sum)
        # print(count)
        # tv_matrix.(sum)
        temp = {'doc_id': i, 'sentTFIDF_score':sum, 'key': sent}
        # print(temp)
        sentTFIDF_scores.append(temp)
    return sentTFIDF_scores


file = open("C:\\Users\\ffayaza\\Documents\\Data\\PROJECT_DATA\\TEST_DATA\\final_data_V1\\dec\\data_03122018.txt", "r",
            encoding="utf-8")
sentences = file.read().split("\n")
test = get_doc(sentences)
# print(test)
freqDict_list = create_freq_dict(sentences)
# print(freqDict_list)
TF_scores = computeTF(test, freqDict_list)
print(TF_scores)
IDF_scores = computeIDF(test, freqDict_list)
TFIDF_scores = computeTFIDF(TF_scores, IDF_scores)
bow = bagofwords(sentences)
tf=computeTFIDFVector(bow,TFIDF_scores,sentences)
print(tf)



# tfidfVector = [computeTFIDFVector(review) for review in tfidfDict]

# sent_TFIDF(TF_scores,IDF_scores,bow)
# tf_idf_model = np.asarray(TFIDF_scores)
#
# tf_idf_model = np.transpose(tf_idf_model)
# print(tf_idf_model)
# sentece_computeTFIDF = sentece_computeTFIDF(TFIDF_scores,test,sentences)
# cosineCalculation(sentece_computeTFIDF)



# similarity_matrix = cosine_similarity(sentece_computeTFIDF['sentTFIDF_score'],sentece_computeTFIDF['sentTFIDF_score'])
# similarity_df = pd.DataFrame(similarity_matrix)
# print(similarity_df)
