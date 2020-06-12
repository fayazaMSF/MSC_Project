from gensim.models import doc2vec
from collections import namedtuple

# Load data
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

doc1 = ["I love machine learning. Its awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]

# Transform data (you can add more data preprocessing steps) 
# tagged_data = [TaggedDocument(words=, tags=[str(i)]) for i, _d in enumerate(data)]
docs = []
analyzedDocument = namedtuple('AnalyzedDocument', ' wordstags')
for i, text in enumerate(doc1):
    print(doc1)
    words = text.lower().split()
    print(words)
    tags = [i]
    print(tags)
    docs.append(analyzedDocument(words, tags))
print("++++++++++++++++++++++")
# Train model (set min_count = 1, if you want the model to work with the provided example data set)
max_epochs = 100
vec_size = 20
alpha = 0.025
model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(analyzedDocument)
print("++++++++++++++++++++++")
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(analyzedDocument,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")
model1 = doc2vec.Doc2Vec(docs, vector_size = 100, window = 300, min_count = 1, workers = 4)

# Get the vectors

print(model1.docvecs[0])
print(model1.corpus_count)

model1.docvecs[1]

from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
test_data = analyzedDocument("I love chatbots".lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['1'])

#loading the model
d2v_model = Doc2Vec.load('doc2vec.model')
#start testing
#printing the vector of document at index 1 in docLabels
docvec = d2v_model.docvecs[1]
print (docvec)
#printing the vector of the file using its name
docvec = d2v_model.docvecs['1.txt'] #if string tag used in training
print(docvec)
#to get most similar document with similarity scores using document-index
similar_doc = d2v_model.docvecs.most_similar(14)
print (similar_doc)
#to get most similar document with similarity scores using document- name
sims = d2v_model.docvecs.most_similar('1.txt')
print (sims)
#to get vector of document that are not present in corpus
docvec = d2v_model.docvecs.infer_vector('war.txt')
print (docvec)