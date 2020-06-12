#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
# reload(sys)
sys.setdefaultencoding('utf-8')
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['text.latex.unicode'] = True
# define training data
sentences = [['கலந்துரையாடல்', 'நிகழ்வு', 'செயலாளரின்', 'தலைமையில்', 'இடம்பெற்றது'],
			['70', 'உறுப்பினர்கள்', 'கலந்து', 'கொண்டனர்'],
			['ிறுகதை', 'புத்தகங்கள்', '10', 'தெரிவு', 'செய்யப்பட்டு', 'அனுப்பப்பட்டுள்ளது']]



text_file = open("D:/Data/test-data/inDomain-pAl-trn.si-ta.si", "r")
lines = text_file.read().decode('utf8').split("\n")
collection = [[]]
for line in lines:
	wordArray = line.split(" ")
	# print wordArray
	collection.append(wordArray)

# train model
# model = Word2Vec(sentences=sentences, size=128, window=5, min_count=5, workers=4)
model = Word2Vec(collection, min_count=1)
# model = Word2Vec(lines, min_count=1)
# summarize the loaded model
# print(model)
# summarize vocabulary
# wordsoriginal = list(model.wv.vocab)
# print(wordsoriginal)
# print (words)
# access vector for one word
# print(model['sentence'])
# save model
# model.save('model.bin')
model.wv.save_word2vec_format('sinhala-inDomain.txt', binary=False)
# load model
# new_model = KeyedVectors.load_word2vec_format('model.txt')
# # new_model = Word2Vec.load('model.txt')
# print(new_model)
# ##############################################
# X = model[model.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
# pyplot.rc('font', family='Arial')
# words = list(new_model.wv.vocab)
# print (words)
# test = u'\u0b89\u0bb1\u0bc1\u0baa\u0bcd\u0baa\u0bbf\u0ba9\u0bb0\u0bcd\u0b95\u0bb3\u0bcd'
# # for i, word in enumerate(words):
# for i, word in enumerate(words):
# 	pyplot.annotate(word.decode('ascii'), xy=(result[i, 0], result[i, 1]))
# 	# pyplot.annotate(word.decode('utf-8'), xy=(result[i, 0], result[i, 1]))
# pyplot.show()