
# from gensim.models import FastText
# import gensim
# import fasttext
# embedding_dict = gensim.models.KeyedVectors.load_word2vec_format("C:/Users/ffayaza/Documents/MscProEmb/WordEmberding/Model/wiki/wiki.ta.vec", binary=False)
# embedding_dict.save_word2vec_format('C:/Users/ffayaza/Documents/MscProEmb/WordEmberding/Model/wiki/saved_model_gensim'+".bin", binary=True)
# model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/ffayaza/Documents/MscProEmb/WordEmberding/Model/wiki/saved_model_gensim'+".bin", binary=True)
# print('தலைமையில்' in model.wv.vocab)
# print('தலைமையில்' in model.wv.vocab)
# print('தலைமையில்' in model.wv.vocab)
# print(model.wv['தலைமையில்'])
# print('end')
#
# # KeyedVectors.load_word2vec_format("C:/Users/ffayaza/Documents/MscProEmb/WordEmberding/Model/wiki/wiki.ta.vec",
# # binary=True, encoding='utf-8', unicode_errors='ignore')
# #
# # import fasttext
# # model = fasttext.load_model('C:/Users/ffayaza/Documents/MscProEmb/WordEmberding/Model/wiki/wiki.ta.vec')
# # print (model['தலைமையில்'])
# -*- encoding: utf8 -*-
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText
import codecs

f_model_file = 'C:/Users/ffayaza/Documents/MscProEmb/WordEmberding/Model/wiki/wiki.ta.vec'
f_model_bin = 'C:/Users/ffayaza/Documents/MscProEmb/WordEmberding/Model/wiki/wiki.ta.bin'
ftext = KeyedVectors.load_word2vec_format(f_model_file)
print(ftext.wv.vocab)
print(ftext.wv['தலைமையில்'])

model = FastText.load_fasttext_format(f_model_bin, encoding='utf8')
print(model.wv.vocab)
print(model.wv['தலைமையில்'])
# model2 = FastText.load_word2vec_format(f_model_file)
# print(model2.wv.vocab)
# print(model2.wv['தலைமையில்'])