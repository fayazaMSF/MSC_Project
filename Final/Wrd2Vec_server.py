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
f_model_bin = '/home/mohan/names/wiki.ta.bin'

v_model_file = '/home/farhath/embeding/models/tamil-all-17-11-24.model'

# '/home/mohan/names/wiki.ta.vec'
# '/home/mohan/names/wiki.ta.bin'
# '/home/farhath/embeding/models/ftext_ti_18-02-07-model.vec'
# '/home/farhath/embeding/models/ftext_ti_18-02-07-model.bin'

# ====================================================
# model = FastText.load_fasttext_format(f_model_bin, encoding='utf8')

ftext_model = KeyedVectors.load_word2vec_format(f_model_file, binary=False,encoding='utf8')

word2vec_model = KeyedVectors.load_word2vec_format(v_model_file, binary=False,encoding='utf8')


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
# .................................................................
def avg_sentence_vector_with_similarity(words, model, num_features, index2word_set):
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
        else:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[model.most_similar(word)])
            print(model.most_similar(word))
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
        # np.mean(featureVec, axis=0)
        # np.divide(featureVec, nwords)
        # print("Num Words", nwords)
        # print("div", featureVec)
    return featureVec
# .................................................
def sent_vectorizer(sent, model):
    sent_vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw += 1
        except:
            pass
    # print(np.asarray(sent_vec) / numw)
    return np.asarray(sent_vec) / numw


# =======================================================================
def cal_sim(sent):
    sim = [0.0] * len(sentences)
    i = 0;
    for sentence in sentences:
        sent_1 = filter(lambda x: x in index2word_set, sent.split("\t")[1].split())
        sent_2 = filter(lambda x: x in index2word_set, sentence.split("\t")[1].split())
        distance = ftext_model.wv.n_similarity(sent_1, sent_2)
        sim[i] = distance
        #        print(distance)
        i += 1
    return sim


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

index2word_set = set(ftext_model.wv.index2word)

file = codecs.open("/home/fayaza/PROJECT_DATA/TEST_DATA/dec/data_03122018.txt", encoding="utf-8")
sentences = file.read().split("\n")
avg_sent_vec = [
    avg_sentence_vector(sent.split("\t")[1].split(), model=ftext_model, num_features=300,
                          index2word_set=index2word_set)
    for sent
    in sentences]
# print(avg_sent_vec)

sim_cal = [cal_sim(news) for news in sentences]
sim_cal_model = np.asarray(sim_cal)
# print(sim_cal_model)


avg_sent_vec_model = np.asarray(avg_sent_vec)

from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(avg_sent_vec_model, avg_sent_vec_model)
# np.matrix(sim_cal_model)
similarity_df = pd.DataFrame(similarity_matrix)
# print(cosine_similarity(tfidfVector[0:1], tfidfVector))
# print(similarity_df)

avg_sent_vec_with_sim = [
    avg_sentence_vector_with_similarity(sent.split("\t")[1].split(), model=ftext_model, num_features=300,
                          index2word_set=index2word_set)
    for sent
    in sentences]

avg_sent_vec_model_with_sim = np.asarray(avg_sent_vec_with_sim)

similarity_matrix_with_sim = cosine_similarity(avg_sent_vec_model_with_sim, avg_sent_vec_model_with_sim)

# =======================================================================
from sklearn.cluster import AffinityPropagation

clustering = AffinityPropagation(affinity='precomputed', copy=True,
                                 damping=0.6, max_iter=200, preference=None, verbose=False).fit_predict(
    similarity_matrix)

clustering_with_sim = AffinityPropagation(affinity='precomputed', copy=True,
                                 damping=0.6, max_iter=200, preference=None, verbose=False).fit_predict(
    similarity_matrix_with_sim)


print(clustering_with_sim)
# ................................................
import numpy as np

mat = np.matrix(similarity_matrix)
from sklearn.cluster import DBSCAN

clustering_db = DBSCAN(min_samples=1, algorithm='auto', eps=0.5, metric='cosine',
                       metric_params=None, n_jobs=1, p=None).fit_predict(avg_sent_vec)

# ================================================

f_out_file = '/home/fayaza/Output/Wrd2Vec/f_result.txt'
f_with_sim_out_file = '/home/fayaza/Output/Wrd2Vec/f_with_sim_result.txt'
write_output(clustering, f_out_file)
write_output(clustering_with_sim, f_with_sim_out_file)
X = []
for sentence in sentences:
    X.append(sent_vectorizer(sentence.split("\t")[1].split(), ftext_model))
f_out_file1 = '/home/fayaza/result4.txt'
write_output(clustering_db, f_out_file1)

# ================================

# from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
# distances = pairwise_distances(avg_sent_vec_model, metric='cosine').astype('float64')
# import hdbscan
# clusterer = HDBSCAN(algorithm='best', approx_min_span_tree=True,
#                    gen_min_span_tree=True, leaf_size=40, metric="precomputed",
#                     min_cluster_size=3, min_samples=None, p=None, core_dist_n_jobs=-1)
#
# clusterer.fit(distances)
# set(clusterer.labels_)
# print(clusterer)


# sentence_1_avg_vector = avg_sentence_vector(sentence_1.split(), model=model, num_features=300, index2word_set=index2word_set)
# sentence_2 = "??? ????? ??????? ???? ???????????? ???? ????? ???"
# sentence_2_avg_vector = avg_sentence_vector(sentence_2.split(), model=model, num_features=300,  index2word_set=index2word_set)
# sim = 1 - spatial.distance.cosine(sentence_1_avg_vector, sentence_2_avg_vector)
# print(sim)
# sen1_sen2_similarity =  cosine_similarity(sentence_1_avg_vector,sentence_2_avg_vector)
