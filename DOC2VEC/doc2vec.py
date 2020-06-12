#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from nltk.tokenize import word_tokenize

# ’s prepare data for training our doc2vec model

corpus = open("C:\\Users\\ffayaza\\Documents\\Data\\MODEL_DATA\\NON_COM_SEN\\data_v4.txt", "r",encoding='utf8')
lines = corpus.read().split("\n")
count = len(lines)
print("Doc Count",count)

#
# data = ["I love machine learning. Its awesome.",
#         "I love coding in python",
#         "I love building chatbots",
#         "they chat amagingly well"]
# for i, _d in enumerate(lines):
#     words = _d.split(" ")
#     tags = [str(i)]
#     print(words)
#     print(tags)
duplicate_dict = {}
for t in lines:
    # print("t===",t)
    if t not in duplicate_dict:
        duplicate_dict[t] = True
        label, news = t.strip().split(' ', 1)
        print("label", label)
        print("text", news)
        fixed =''.join([x if x.isalnum() or x.isspace() else " " for x in news ]).split()
tagged_data = [TaggedDocument(words=" ".join(_d.split()).split(' ', 1)[1].split(" "), tags=[str(_d.split(' ', 1)[0])]) for i, _d in enumerate(lines)]
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
model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)
# dm=1 means ‘distributed memory’ (PV-DM) and dm =0 means ‘distributed bag of words’ (PV-DBOW).
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v2.model")
print("Model Saved")

from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load("d2v2.model")
#to find the vector of a document which is not in training data
# test_data = word_tokenize("I love chatbots".lower())
# v1 = model.infer_vector(test_data)
# print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('140572')
print(similar_doc)
#
#
# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['140572'].tostring().encoding("utf8"))

# for doc in docs:
#     doc_vecs = model.infer_vector(doc.split())
# # creating a matrix from list of vectors
# mat = np.stack(doc_vecs)
#
# # Clustering Kmeans
# km_model = KMeans(n_clusters=5)
# km_model.fit(mat)
# # Get cluster assignment labels
# labels = km_model.labels_
#
# # Clustering DBScan
# dbscan_model = DBSCAN()
# labels = dbscan_model.fit_predict(mat)