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
# C:\\Users\\ffayaza\\Documents\\Data\\PROJECT_DATA\\TEST_DATA\\final_data_V1\\nov\\%s.txt
sentences = file.read().split("\n")
tfDict = [computeReviewTFDict(sent) for sent in sentences]
# print(tfDict)

import xml.etree.ElementTree as ET

path = "C:\\Users\\ffayaza\\Documents\\testData\\tokenized\\%s.xml" %date
# C:\\Users\\ffayaza\\Documents\\Data\\PROJECT_DATA\\TEST_DATA\\tokenized_data\\nov\\%s.xml
tree = ET.parse(path)
root = tree.getroot()
news = [news for news in tree.findall('NEWS')]

f_out_file="C:\\Users\\ffayaza\\Documents\\Output\\Title\\tfidf\\basic\\%s.xml" %date
# C:\\Users\\ffayaza\\Documents\\Output\\Tf_Idf\\basic\\nov\\%s.xml
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
# print(cosine_similarity(tfidfVector[0:1], tfidfVector))
print(similarity_df)
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



text_file = open(f_out_file, "w",encoding='utf8')
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

