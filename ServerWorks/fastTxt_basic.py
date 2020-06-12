import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText
from scipy import spatial
import pandas as pd
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from gensim.models.keyedvectors import KeyedVectors
import codecs

f_model_file = '/home/mohan/names/wiki.ta.vec'
#'/home/fayaza/Model/FastText/ftext_ta_19-08-28-model.vec'
f_model_bin ='/home/fayaza/Model/FastText/ftext_cbow_ta_19-08-28-model.bin'


# ====================================================
# model = FastText.load_fasttext_format(f_model_bin, encoding='utf8')

ftext_model = KeyedVectors.load_word2vec_format(f_model_file, binary=False,encoding='utf8')

# ==============================================================

def avg_sentence_vector(words, model, num_features, index2word_set):
    # function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        # print(word)
        if word in index2word_set:
            # print(word)
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])
            # print("added",featureVec)

    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
        # np.mean(featureVec, axis=0)
        # np.divide(featureVec, nwords)
        # print("Num Words", nwords)
        # print("div", featureVec)
    return featureVec


# ========================================================
date="03122018"
index2word_set = set(ftext_model.wv.index2word)

file = codecs.open("/home/fayaza/PROJECT_DATA/TEST_DATA/dec/data_%s.txt" %date, encoding="utf-8")
sentences = file.read().split("\n")
avg_sent_vec = [
    avg_sentence_vector(sent.split("\t")[1].split(), model=ftext_model, num_features=300,
                          index2word_set=index2word_set)
    for sent
    in sentences]

avg_sent_vec_model = np.asarray(avg_sent_vec)

from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(avg_sent_vec_model, avg_sent_vec_model)
similarity_df = pd.DataFrame(similarity_matrix)
print(similarity_df)
# =======================================================================

import xml.etree.ElementTree as ET

path = "/home/fayaza/PROJECT_DATA/tokenized_data/dec/data_%s.xml" %date
tree = ET.parse(path)
root = tree.getroot()
news = [news for news in tree.findall('NEWS')]

f_out_file="/home/fayaza/Output/Test/wiki/basic/oct/data_%s.xml" %date

cluster_List = []
visited_List = []
x = 0
for i, row in similarity_df.iterrows():
    clst_dict = []
    if (i not in visited_List):
        for j, column in row.iteritems():
            if (j not in visited_List):
                if (column >0.15):
                    visited_List.append(j)
                    clst_dict.append(j+1)
                # print(i, j)
            # clst_dict[j]+=1
        temp = { x: clst_dict}
        cluster_List.append(temp)
        x+=1
print(cluster_List)



text_file =codecs.open(f_out_file, "w",encoding='utf8')
text_file.write("<ITEMS>" + "\n")
a=0
for cluster in cluster_List:
    for k,v in cluster.items():

        for i in v:
            # print(i)
            cls_news=news[i-1]
            print(sentences[i-1])
            # text_file.write(str(a) + "\t")
            # text_file.write(str(cls_news.find('TITLE').text))
            text_file.write("<NEWS> " + "\n")
            text_file.write("<CLUSTER> " + str(a) + " </CLUSTER>")
            text_file.write("\n" + "<LINK> " + str(cls_news.find('LINK').text) + " </LINK>" + "\n")
            text_file.write("<TITLE> " + str(cls_news.find('TITLE').text) + " </TITLE>" + "\n")
            text_file.write("<BODY> " + str(cls_news.find('BODY').text) + " </BODY>" + "\n")
            text_file.write("<DATE> " + str(cls_news.find('DATE').text) + " </DATE>" + "\n")
            text_file.write("</NEWS> " + "\n")
            text_file.write("\n")
        a+=1
text_file.write("</ITEMS>" + "\n")
text_file.close()
print(visited_List)


# ============================Write =====
import operator
def write_output(clustering, file_name):
    print ('print files ')
    text_file = open(file_name, "w",encoding='utf8')
    cluster_news = []
    i=0
    for sent in sentences:
        temp = {'cluster': clustering[i],'news':sent.split("\t")[1]}
        cluster_news.append(temp)
        i+=1
    cluster_news=(sorted(cluster_news, key=lambda i: i['cluster']))
    # return cluster_news
    # text_file.write("---------------- " + word + "    ----------------------- \n")
    for news in cluster_news:
        # text_file.write(str(news.keys()))
        text_file.write(str(news['cluster'])+"\t")
        text_file.write(str(news['news']))
        text_file.write("\n")
    text_file.close()

