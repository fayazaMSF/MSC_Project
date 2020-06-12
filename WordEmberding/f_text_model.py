#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fasttext

# Skipgram model
# model = fasttext.skipgram('data.txt', 'model')
# print model.words # list of words in dictionary
#
# # CBOW model
# model = fasttext.cbow('data.txt', 'model')
# print model.words # list of words in dictionary


def create_skipgram_model():
    print ('about to create skipgram moel')
    model = fasttext.skipgram('/home/farhath/embeding/data/all-18-02-07.ta', '/home/farhath/embeding/models/ftext_ti_18-02-07-model')
    print ('done skip')
    print (model.words ) # list of words in dictionary


create_skipgram_model()