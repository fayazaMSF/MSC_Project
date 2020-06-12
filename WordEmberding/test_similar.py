#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot
from matplotlib import rcParams

rcParams['text.latex.unicode'] = True

# load model
model = KeyedVectors.load_word2vec_format('sinhala-parallel.txt')
vocab = model.vocab
vocab2 = model.wv.vocab

similar1 = model.similarity(u'පාසල්', u'පාසල්වල')
similar2 = model.similarity(u'පාර්ලිමේන්තුව', u'පාර්ලිමේන්තුවේ')
similar3 = model.similarity(u'පාර්ලිමේන්තුව', u'පාර්ලිමේන්තුව')
print(similar1)
print(similar2)
print(similar3)
