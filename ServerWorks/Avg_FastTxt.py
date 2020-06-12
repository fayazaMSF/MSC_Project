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
f_model_bin ='/home/fayaza/Model/FastText/ftext_cbow_ta_19-08-28-model.bin'
#/home/fayaza/Model/FastText/cus/ftext_ta_19-09-25-model.bin
#/home/fayaza/Model/FastText/ftext_cbow_ta_19-08-28-model.bin
#ftext_ta_19-08-28-model.bin'
#'/home/farhath/embeding/models/ftext_ti_18-02-07-model.bin'
# '/home/mohan/names/wiki.ta.bin'

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

# =======================================================================
def write_output(clustering, file_name):
    print('print files ')
    text_file = open(file_name, "w")
    cluster_news = []
    i = 0
    for sent in sentences:
        temp = {'cluster': clustering[i], 'news': sent.split("\t")[1]}
        cluster_news.append(temp)
        i += 1
    cluster_news = (sorted(cluster_news, key=lambda i: i['cluster']))
    # return cluster_news
    # text_file.write("---------------- " + word + "    ----------------------- \n")
    for news in cluster_news:
        # text_file.write(str(news.keys()))
        text_file.write(str(news['cluster']) + "\t")
        text_file.write(str(news['news']))
        text_file.write("\n")
    text_file.close()


# ========================================================

index2word_set = set(ftext_model.wv.index2word)

file = codecs.open("/home/fayaza/PROJECT_DATA/TEST_DATA/dec/data_01122018.txt", encoding="utf-8")
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
# print(cosine_similarity(tfidfVector[0:1], tfidfVector))
# print(similarity_df)

# =======================================================================
from sklearn.cluster import AffinityPropagation

clustering = AffinityPropagation(affinity='precomputed', copy=True,
                                 damping=0.6, max_iter=200, preference=None, verbose=False).fit_predict(
    similarity_matrix)

# ................................................
import numpy as np

mat = np.matrix(similarity_matrix)
from sklearn.cluster import DBSCAN

clustering_db = DBSCAN(min_samples=1, algorithm='auto', eps=0.5, metric='cosine',
                       metric_params=None, n_jobs=1, p=None).fit_predict(avg_sent_vec)

# ================================================

f_out_file_affi = '/home/fayaza/Output/fasttext/01122018_f_result_Affi_v1.txt'

write_output(clustering, f_out_file_affi)
f_out_file_db = '/home/fayaza/Output/fasttext/01122018_f_result_DB.txt'
write_output(clustering_db, f_out_file_db)

# ================================
