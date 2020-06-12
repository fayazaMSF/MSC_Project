from __future__ import print_function
from gensim.models import KeyedVectors

# Creating the model
ta_model = KeyedVectors.load_word2vec_format('C:\\Users\\ffayaza\\Documents\\MscProject\\WordEmberding\\Model\\wiki\\wiki.ta.vec')

# Getting the tokens
words = []
for word in ta_model.vocab:
    words.append(word)

# Printing out number of tokens available
print("Number of Tokens: {}".format(len(words)))

# Printing out the dimension of a word vector
print("Dimension of a word vector: {}".format(
    len(ta_model[words[0]])
))

# Print out the vector of a word
print("Vector components of a word: {}".format(
    ta_model[words[0]]
))
print("Vecort value for வெள்ளம்")
print(ta_model.wv.get_vector('வெள்ளம்'))

# Pick a word
find_similar_to = 'வெள்ளம்'

# Finding out similar words [default= top 10]
for similar_word in ta_model.similar_by_word(find_similar_to):
    print("Word: {0}, Similarity: {1:.2f}".format(
        similar_word[0], similar_word[1]
    ))

# Output
# Word: cars, Similarity: 0.83
# Word: automobile, Similarity: 0.72
# Word: truck, Similarity: 0.71
# Word: motorcar, Similarity: 0.70
# Word: vehicle, Similarity: 0.70
# Word: driver, Similarity: 0.69
# Word: drivecar, Similarity: 0.69
# Word: minivan, Similarity: 0.67
# Word: roadster, Similarity: 0.67
# Word: racecars, Similarity: 0.67

# Test words
word_add = ['இலங்கை', 'இந்தியா']
word_sub = ['கேரள']

# Word vector addition and subtraction
for resultant_word in ta_model.most_similar(
    positive=word_add, negative=word_sub
):
    print("Word : {0} , Similarity: {1:.2f}".format(
        resultant_word[0], resultant_word[1]
    ))

# Output

# Word : delhi , Similarity: 0.77
# Word : indore , Similarity: 0.76
# Word : bangalore , Similarity: 0.75
# Word : mumbai , Similarity: 0.75
# Word : kolkata , Similarity: 0.75
# Word : calcutta,india , Similarity: 0.75
# Word : ahmedabad , Similarity: 0.75
# Word : pune , Similarity: 0.74
# Word : kolkata,india , Similarity: 0.74
# Word : kolkatta , Similarity: 0.74
