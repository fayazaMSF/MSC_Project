# Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import doc2vec
from collections import namedtuple

corpus = open("/home/fayaza/PROJECT_DATA/allData-19-08-27.ta", "r", encoding='utf8')
lines = corpus.read().split("\n")
count = len(lines)
print("Doc Count", count)

def tokenize_text(text):
    tokens = []
    text=" ".join(text.split())
    for word in text.split(" "):
        if  word.isspace():
            # print(">>>>>>>>>>>")
            # print(word)
            continue
        else:
            # print("-------------")
            # print(word)
            tokens.append(word)
    return tokens


tagged_data = [TaggedDocument(words=tokenize_text(doc), tags=[i]) for i, doc in enumerate(lines)]
print(tagged_data)

max_epochs = 100
vec_size = 300
alpha = 0.025

# simple_models = [
#     # PV-DBOW plain
#     Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0,
#             epochs=20, workers=cores),
#     # PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
#     Doc2Vec(dm=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, sample=0,
#             epochs=20, workers=cores, alpha=0.05, comment='alpha=0.05'),
#     # PV-DM w/ concatenation - big, slow, experimental mode
#     # window=5 (both sides) approximates paper's apparent 10-word total window size
#     Doc2Vec(dm=1, dm_concat=1, vector_size=100, window=5, negative=5, hs=0, min_count=2, sample=0,
#             epochs=20, workers=cores),
# ]
# model = Doc2Vec(size=vec_size,
#                 alpha=alpha,
#                 min_alpha=0.00025,
#                 min_count=2,
#                 dm=1)
#
# model.build_vocab(tagged_data)
# # dm=1 means ‘distributed memory’ (PV-DM) and dm =0 means ‘distributed bag of words’ (PV-DBOW).
# for epoch in range(max_epochs):
#     print('iteration {0}'.format(epoch))
#     model.train(tagged_data,
#                 total_examples=model.corpus_count,
#                 epochs=model.iter)
#     # decrease the learning rate
#     model.alpha -= 0.0002
#     # fix the learning rate, no decay
#     model.min_alpha = model.alpha
#
# model.save("/home/fayaza/Model/Doc2Vec/Doc2Vec_PV_DM.model")
# print("Model Saved")
