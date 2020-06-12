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

v_model_file = '/home/farhath/embeding/models/tamil-all-17-11-24.model'

# ====================================================

word2vec_model = KeyedVectors.load_word2vec_format(v_model_file, binary=False,encoding='utf8')

# ==============================================================
list_item = []
def avg_sentence_vector(words, num_features):
    # function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    word2vec = KeyedVectors.load_word2vec_format(v_model_file)
    embed_vocab = word2vec.wv.vocab

    for word in words:
        if (word in embed_vocab):
            nwords = nwords + 1
            featureVec = np.add(featureVec, word2vec[word])

    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

# ============================================================
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

file = codecs.open("/home/fayaza/PROJECT_DATA/TEST_DATA/dec/data_03122018.txt", encoding="utf-8")
sentences = file.read().split("\n")
avg_sent_vec = [
    avg_sentence_vector(sent.split("\t")[1].split(),num_features=100)
    for sent
    in sentences]
# print(avg_sent_vec)

avg_sent_vec_model = np.asarray(avg_sent_vec)

from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(avg_sent_vec_model, avg_sent_vec_model)
# np.matrix(sim_cal_model)
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

W_out_file_affi = '/home/fayaza/Output/Wrd2Vec/03122018_w_result_Affi.txt'
write_output(clustering, W_out_file_affi)

W_out_file_db = '/home/fayaza/Output/Wrd2Vec/03122018_w_result_DB.txt'
write_output(clustering_db, W_out_file_db)
