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

f_model_file = '/home/fayaza/Model/FastText/ftext_ta_19-08-28-model.vec'
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
date="31102018"
index2word_set = set(ftext_model.wv.index2word)

file = codecs.open("/home/fayaza/PROJECT_DATA/TEST_DATA/oct/data_%s.txt" %date, encoding="utf-8")
sentences = file.read().split("\n")
avg_sent_vec = [
    avg_sentence_vector(sent.split("\t")[1].split(), model=ftext_model, num_features=100,
                          index2word_set=index2word_set)
    for sent
    in sentences]

avg_sent_vec_model = np.asarray(avg_sent_vec)

from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(avg_sent_vec_model, avg_sent_vec_model)
similarity_df = pd.DataFrame(similarity_matrix)

# =======================================================================
from sklearn.cluster import AffinityPropagation

clustering = AffinityPropagation(preference=-50).fit(similarity_matrix)
clustering = AffinityPropagation(affinity='precomputed', copy=True,
                                 damping=0.6, max_iter=200, verbose=False).fit_predict(
    similarity_matrix)

# ================================================

# ================================
import xml.etree.ElementTree as ET

path = "/home/fayaza/PROJECT_DATA/tokenized_data/oct/data_%s.xml" %date
tree = ET.parse(path)
root = tree.getroot()
cl_news = [news for news in tree.findall('NEWS')]


def write_output(clustering, file_name,cl_news):
    print ('print files ')
    text_file = codecs.open(file_name, "w",encoding='utf8')
    text_file.write("<ITEMS>" + "\n")
    cluster_news = []
    i=0
    for sent in sentences:
        temp = {clustering[i]: cl_news[i]}
        # temp = {'cluster': clustering[i], 'news': cl_news[i]}
        cluster_news.append(temp)
        i+=1
    # cluster_news = (sorted(cluster_news, key=lambda i: i['cluster']))
    cluster_news =  sorted(cluster_news, key=lambda d: list(d.keys()))
    # print(cluster_news)
        # (sorted(cluster_news.keys()))
    # return cluster_news
    # text_file.write("---------------- " + word + "    ----------------------- \n")
    a=0
    for cluster in cluster_news:
        for v in cluster.items():
            # print(k)
            # print(v)
            # for x in v:
            #     print(i)
                # cls_news = cl_news[a]
                # print(sentences[i - 1])
                # text_file.write(str(a) + "\t")
                # text_file.write(str(cls_news.find('TITLE').text))
            text_file.write("<NEWS> " + "\n")
            text_file.write("<CLUSTER> " + str(v[0]) + " </CLUSTER>")
            text_file.write("\n" + "<LINK> " + str(v[1].find('LINK').text) + " </LINK>" + "\n")
            text_file.write("<TITLE> " + str(v[1].find('TITLE').text) + " </TITLE>" + "\n")
            text_file.write("<BODY> " + str(v[1].find('BODY').text) + " </BODY>" + "\n")
            text_file.write("<DATE> " + str(v[1].find('DATE').text) + " </DATE>" + "\n")
            text_file.write("</NEWS> " + "\n")
            text_file.write("\n")
            a += 1
    text_file.write("</ITEMS>" + "\n")
    text_file.close()
    # for news in cluster_news:
    #     text_file.write("<NEWS> " + "\n")
    #     text_file.write("<CLUSTER> " + str(a) + " </CLUSTER>")
    #     text_file.write("\n" + "<LINK> " + str(news.find('LINK').text) + " </LINK>" + "\n")
    #     text_file.write("<TITLE> " + str(news.find('TITLE').text) + " </TITLE>" + "\n")
    #     text_file.write("<BODY> " + str(news.find('BODY').text) + " </BODY>" + "\n")
    #     text_file.write("<DATE> " + str(news.find('DATE').text) + " </DATE>" + "\n")
    #     text_file.write("</NEWS> " + "\n")
    #     text_file.write("\n")
    #     a+=1
    # text_file.close()
f_out_file="/home/fayaza/Output/Test/wiki/affinity/oct/data_%s.xml" %date

write_output(clustering, f_out_file,cl_news)
# print(cluster_news)
# print(sorted(cluster_news, key=lambda i: i['cluster']))
#
# f_out_file_affi = '/home/fayaza/Output/fasttext/31102018_f_result_Affi.txt'
#
# write_output(clustering, f_out_file_affi)
# f_out_file_db = '/home/fayaza/Output/fasttext/01122018_f_result_DB.txt'
# write_output(clustering_db, f_out_file_db)
