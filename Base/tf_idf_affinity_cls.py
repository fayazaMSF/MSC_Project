import math
import numpy as np
import pandas as pd
from string import digits
remove_digits = str.maketrans('', '', digits)
def computeReviewTFDict(review):
    """ Returns a tf dictionary for each review whose keys are all
    the unique words in the review and whose values are their
    corresponding tf.
    """
    # print(review)
    review = review.split("\t")[1].translate(remove_digits).split()
    #Counts the number of times the word appears in review
    reviewTFDict = {}
    for word in review:
        if word in reviewTFDict:
            reviewTFDict[word] += 1
        else:
            reviewTFDict[word] = 1
    #Computes tf for each word
    for word in reviewTFDict:
        reviewTFDict[word] = reviewTFDict[word] / len(review)
    return reviewTFDict
date="data_30112018"
file = open("C:\\Users\\ffayaza\\Documents\\Data\\PROJECT_DATA\\TEST_DATA\\Title_final_data\\%s.txt" %date, "r",
            encoding="utf-8")
# C:\\Users\\ffayaza\\Documents\\Data\\PROJECT_DATA\\TEST_DATA\\final_data_V1\\dec\\data_%s.txt
sentences = file.read().split("\n")
tfDict = [computeReviewTFDict(sent) for sent in sentences]
# print(tfDict)

def computeCountDict():
    """ Returns a dictionary whose keys are all the unique words in
    the dataset and whose values count the number of reviews in which
    the word appears.
    """
    countDict = {}
    # Run through each review's tf dictionary and increment countDict's (word, doc) pair
    for review in tfDict:
        # print(review)
        for word in review:
            if word in countDict:
                countDict[word] += 1
            else:
                countDict[word] = 1
    return countDict

countDict = computeCountDict()
# print(countDict)
# print(len(countDict))
# print(len(sentences))

def computeIDFDict():
    """ Returns a dictionary whose keys are all the unique words in the
    dataset and whose values are their corresponding idf.
    """
    idfDict = {}
    for word in countDict:
        idfDict[word] = math.log(len(sentences) / countDict[word])
    return idfDict

idfDict = computeIDFDict()
# print(idfDict)

def computeReviewTFIDFDict(reviewTFDict):
    """ Returns a dictionary whose keys are all the unique words in the
    review and whose values are their corresponding tfidf.
    """
    # print(reviewTFDict)
    # print("_____________________")
    reviewTFIDFDict = {}
    #For each word in the review, we multiply its tf and its idf.
    for word in reviewTFDict:
        # print(word+"--")
        reviewTFIDFDict[word] = reviewTFDict[word] * idfDict[word]
        # print(reviewTFDict[word])
        # print(idfDict[word])
        # print(reviewTFIDFDict[word])

    return reviewTFIDFDict

tfidfDict = [computeReviewTFIDFDict(review) for review in tfDict]
# print(tfidfDict)

## Create a list of unique words
wordDict = sorted(countDict.keys())
# print(wordDict)

def computeTFIDFVector(review):
      tfidfVector = [0.0] * len(wordDict)
      # print(review)
      # For each unique word, if it is in the review, store its TF-IDF value.
      for i, word in enumerate(wordDict):
          if word in review:
              # print("____________"+word)
              # print(review[word])
              # print(i)
              # print("____________" + str(review[word]))
              tfidfVector[i] = review[word]
      return tfidfVector

tfidfVector = [computeTFIDFVector(review) for review in tfidfDict]
# print(tfidfVector)
tf_idf_model = np.asarray(tfidfVector)
# print(tf_idf_model)
# print(tf_idf_model.shape)
# =============================================
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tf_idf_model, tf_idf_model)
# print(similarity_matrix)
similarity_df = pd.DataFrame(similarity_matrix)
# similarity_df.sort_values()
# print(cosine_similarity(tfidfVector[0:1], tfidfVector))
# print(similarity_df)

# from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
#
# distances = pairwise_distances(tf_idf_model, metric='cosine').astype('float64')
# import hdbscan
# clusterer = HDBSCAN(algorithm='best', approx_min_span_tree=True,
#                     gen_min_span_tree=True, leaf_size=40, metric="precomputed",
#                     min_cluster_size=3, min_samples=None, p=None, core_dist_n_jobs=-1)
#
# clusterer.fit(distances)s
# set(clusterer.labels_)
# print(clusterer)

# =======================================================================
# import numpy as np
# from sklearn.cluster import SpectralClustering
# mat = np.matrix(similarity_matrix)
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import pairwise_distances
# # X = StandardScaler().fit_transform(tfidfVector)
# # print(X)
# print((DBSCAN(min_samples=1,algorithm='auto', eps=0.88, metric='cosine',
#     metric_params=None, n_jobs=1, p=None).fit_predict(tfidfVector)))
# ===================================pairwise_distances(tfidfVector,metric='cosine')
from sklearn.cluster import AffinityPropagation
clustering = AffinityPropagation(affinity='precomputed', copy=True,
          damping=0.75, max_iter=200).fit_predict(similarity_matrix)


# print(clustering)
# # ===================================
# from sklearn.cluster import MeanShift
# ms = MeanShift()
# # ms.fit(tf_idf_model)
# print(ms.fit_predict(tf_idf_model))
# # =======================================
# # print(SpectralClustering(affinity='rbf', assign_labels='kmeans', coef0=1, degree=3,
# #           eigen_solver=None, eigen_tol=0.0, gamma=1.0, kernel_params=None,
# #           n_clusters=6, n_init=10, n_jobs=1, n_neighbors=10,
# #           random_state=None).fit_predict(similarity_matrix))
#
# # =====================
# # import hierarchical clustering libraries
# import scipy.cluster.hierarchy as sch
# from sklearn.cluster import AgglomerativeClustering
# # create dendrogram
# import matplotlib.pyplot as plt
# dendrogram = sch.dendrogram(sch.linkage(tf_idf_model, method='ward'))
# # ========================================
# # create clusters
# hc = AgglomerativeClustering(affinity = 'euclidean', linkage = 'ward')
# # save clusters for chart
# y_hc = hc.fit_predict(tf_idf_model)
# print(y_hc)
# plt.scatter(tf_idf_model[y_hc ==0,0], tf_idf_model[y_hc == 0,1], s=100, c='red')
# plt.scatter(tf_idf_model[y_hc==1,0], tf_idf_model[y_hc == 1,1], s=100, c='black')
# plt.scatter(tf_idf_model[y_hc ==2,0], tf_idf_model[y_hc == 2,1], s=100, c='blue')
# plt.scatter(tf_idf_model[y_hc ==3,0], tf_idf_model[y_hc == 3,1], s=100, c='cyan')
# plt.show()
# # ======================================
# # from scipy.cluster.hierarchy import dendrogram, linkage
# #
# # Z = linkage(similarity_matrix, 'ward')
# # print(pd.DataFrame(Z, columns=['Document\Cluster 1', 'Document\Cluster 2',
# #                          'Distance', 'Cluster Size'], dtype='object'))
# #
# # plt.figure(figsize=(8, 3))
# # plt.title('Hierarchical Clustering Dendrogram')
# # plt.xlabel('Data point')
# # plt.ylabel('Distance')
# # dendrogram(Z)
# # plt.axhline(y=1.0, c='k', ls='--', lw=0.5)
# # plt.show()
# # ===========================================
# # import hdbscan
#
# # from scipy.cluster import  hierarchy
# # X = tf_idf_model.todense()
# # threshold = 0.1
# # Z = hierarchy.linkage(X,"average", metric="cosine")
# # C = hierarchy.fcluster(Z, threshold, criterion="distance")
# # #
#
#
# # ========================================================================
# # def dot_product(vector_x, vector_y):
# #     dot = 0.0
# #     for e_x, e_y in zip(vector_x, vector_y):
# #         dot += e_x * e_y
# #     return dot
# #
# # def magnitude(vector):
# #     mag = 0.0
# #     for index in vector:
# #       mag += math.pow(index, 2)
# #     return math.sqrt(mag)
# #
# # review_similarity = dot_product(tfidfVector[1], tfidfVector[1])/ magnitude(tfidfVector[1]) * magnitude(tfidfVector[1])
# # # print(review_similarity)
# ============================================================
# ============================Write =====
import operator
import xml.etree.ElementTree as ET

path = "C:\\Users\\ffayaza\\Documents\\testData\\tokenized\\%s.xml" %date
# C:\\Users\\ffayaza\\Documents\\Data\\PROJECT_DATA\\TEST_DATA\\tokenized_data\\dec\\data_%s.xml
tree = ET.parse(path)
root = tree.getroot()
cl_news = [news for news in tree.findall('NEWS')]

# f_out_file="C:\\Users\\ffayaza\\Documents\\Output\\Tf_Idf\\basic\\oct\\data_31102018.xml"

def write_output(clustering, file_name,cl_news):
    print ('print files ')
    text_file = open(file_name, "w",encoding='utf8')
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
f_out_file="C:\\Users\\ffayaza\\Documents\\Output\\Title\\tfidf\\affinity\\%s.xml" %date
# C:\\Users\\ffayaza\\Documents\\Output\\Tf_Idf\\affinity\\dec\\data_%s.xml
write_output(clustering, f_out_file,cl_news)
# print(cluster_news)
# print(sorted(cluster_news, key=lambda i: i['cluster']))


