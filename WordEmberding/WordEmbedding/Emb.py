
# https://datascience.stackexchange.com/questions/20071/how-do-i-load-fasttext-pretrained-model-with-gensim
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText
from scipy import spatial
import codecs
import xml.etree.ElementTree as ET
#
# path = "C:\\Users\\ffayaza\\Documents\\MscProject\\Data\\sampl.xml"
# tree = ET.parse(path)
# root = tree.getroot()
#
# # list so that we don't mess up the order of iteration when removing items.
# iterator = list(root.getiterator('NEWS'))
#
# for item in iterator:
#     title = item.find('TITLE')
#     Title_text = title.text
#     print(Title_text)
#     body = item.find('BODY')
#     body_text = body.text
#     print(body_text)

f_model_file = 'C:/Users/ffayaza/Documents/MscProEmb/WordEmberding/Model/wiki/wiki.ta.vec'
f_model_bin = 'C:/Users/ffayaza/Documents/MscProEmb/WordEmberding/Model/wiki/wiki.ta.bin'
ftext = KeyedVectors.load_word2vec_format(f_model_file)
# print(ftext.wv.vocab)
# print(ftext.wv['தலைமையில்'])

model = FastText.load_fasttext_format(f_model_bin, encoding='utf8')
# print(model.wv.vocab)
# print(model.wv['தலைமையில்'])

def avg_sentence_vector(words, model, num_features, index2word_set):
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        print(word)
        if word in index2word_set:
            print(word)
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])
            print("added",featureVec)

    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
        print("Num Words", nwords)
        print("div", featureVec)
    return featureVec
index2word_set = set(model.wv.index2word)
#get average vector for sentence 1
sentence_1 = "கஜா சூறாவளி அடுத்த 12 மணித்தியாலங்களில் மேலும் வலுவடையக்கூடும்"
# print(title)
sentence_1_avg_vector = avg_sentence_vector(sentence_1.split(), model=model, num_features=300, index2word_set=index2word_set)
print(sentence_1_avg_vector)
#get average vector for sentence 2
sentence_2 = "கஜா புயல் காரணமாக யாழ் குடாநாட்டில் நாளை கடும் மழை"
sentence_2_avg_vector = avg_sentence_vector(sentence_2.split(), model=model, num_features=300,  index2word_set=index2word_set)
sim = 1 - spatial.distance.cosine(sentence_1_avg_vector, sentence_2_avg_vector)
print(sim)
# sen1_sen2_similarity =  cosine_similarity(sentence_1_avg_vector,sentence_2_avg_vector)