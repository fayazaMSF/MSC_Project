#!/usr/bin/env python
# -*- encoding: utf8 -*-
import sys
sys.setdefaultencoding('utf-8')
from gensim.models.keyedvectors import KeyedVectors
import codecs


list_file = '/home/farhath/embeding/data/test-list.si'
f_model_file = '/home/farhath/embeding/models/ftext_si_18-02-07-model.vec'
v_model_file = '/home/farhath/embeding/models/sinhala-all-17-12-20.model'
# ftext_ti_18-02-07-model.bin
# ftext_ti_18-02-07-model.vec
list_item = []
v_similars = {}
f_similars = {}

v_out_file ='/home/farhath/embeding/result/v_test_out.si'
f_out_file = '/home/farhath/embeding/result/f_test_out.si'

def get_word_list():
    print('open the file and get list of words ..')
    for line in codecs.open(list_file, encoding='utf8'):
        list_item.append(line.strip())

def write_output(result, file_name):
    print ('print files ')
    text_file = open(file_name, "w")

    for word, mappings in result.iteritems():
        text_file.write("---------------- " + word + "    ----------------------- \n")
        for word in mappings :
            text_file.write(word[0] + " - ")
            text_file.write(str(word[1]))
            text_file.write("\n")
    text_file.close()

def ftxt_similara():
    ftext = KeyedVectors.load_word2vec_format(f_model_file)
    embed_vocab = ftext.wv.vocab
    for word in list_item:
        similar = []
        if (word in embed_vocab):
            similar = ftext.most_similar(word)
        # similar = word2vec.most_similar(word, topn=1)
        f_similars[word] = similar

def w2vec_similars():
    word2vec = KeyedVectors.load_word2vec_format(v_model_file)
    embed_vocab = word2vec.wv.vocab

    for word in list_item:
        similar = []
        if (word in embed_vocab):
            similar = word2vec.most_similar(word)
            # similar = word2vec.most_similar(word, topn=1)
        v_similars[word] = similar

get_word_list()
ftxt_similara()
write_output(f_similars, f_out_file)

w2vec_similars()
write_output(v_similars, v_out_file)