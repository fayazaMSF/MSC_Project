import logging
import re
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def preprocess(str):
    # remove links
    str = re.sub(r'http(s)?:\/\/\S*? ', "", str)

    return str

class Documents(object):
    def __init__(self, documents):
        self.documents = documents

    def __iter__(self):
        for i, doc in enumerate(self.documents):
            yield TaggedDocument(words = doc, tags = [i])
file = "cyber-trend-index-dataset-version2.txt"
corpus = open("C:\\Users\\ffayaza\\Documents\\Data\\MODEL_DATA\\NON_COM_SEN\\data_v4.txt", "r",encoding='utf8')
lines = corpus.read().split("\n")
count = len(lines)
preprocessed = []

duplicate_dict = {}

for t in lines:
    if t not in duplicate_dict:
        duplicate_dict[t] = True
        # t = preprocess(t)
        fixed =''.join([x if x.isalnum() or x.isspace() else " " for x in t ]).split()
        preprocessed.append(fixed)

documents = Documents(preprocessed)
print(documents)



# #iter = 1, because we keep training ourselves :)
# model = Doc2Vec(size=100, dbow_words= 1, dm=0, iter=1,  window=5, seed=1337, min_count=5, workers=4,alpha=0.025, min_alpha=0.025)
# model.build_vocab(documents)
# for epoch in range(10):
#     print("epoch "+str(epoch))
#     model.train(documents, total_examples=count, epochs=1)
#     model.save('news.model')
#     model.alpha -= 0.002  # decrease the learning rate
#     model.min_alpha = model.alpha  # fix the learning rate, no decay
