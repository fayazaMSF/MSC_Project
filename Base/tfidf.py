#!/usr/bin/python
import numpy as np

def tokenize_sentences(sentences):
    words = []
    for sentence in sentences:
        word = sentence.split("\t")[1].split()
        # print(sentence_)
        # w = extract_words(sentence_)
        # print(word)
        words.extend(word)
        # print(len(word))
    # print(">>>>>>>>")
    print(len(words))
    output = set()
    for x in words:
        output.add(x)
    # print(output)
    words =list(set(words))
    # print(words)
    print(len(words))
    return words


def extract_words(sentence):
    for sent in sentence:
        words = sent.split("\t")[1].split()
        print(words)
        words_cleaned = [w.strip() for w in words]
    return words_cleaned


def bagofwords(sentence, words):
    sentence_words = extract_words(sentence)
    # frequency word count
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i, word in enumerate(words):
            if word == sw:
                bag[i] += 1

    return np.array(bag)


file = open("C:\\Users\\ffayaza\\Documents\\Data\\PROJECT_DATA\\TEST_DATA\\final_data_V1\\dec\\data_03122018.txt", "r",
            encoding="utf-8")

sentences = file.read().split("\n")
print(sentences)
count = len(sentences)
print(count)
word = tokenize_sentences(sentences)
bg = bagofwords(sentences,word)
print(bg)
tf_idf_model = np.asarray(bg)
print(tf_idf_model)
#
# vocabulary = tokenize_sentences(sentences)
# # print(vocabulary)
# # bagofwords("ரஞ்சனுக்கு நீதிமன்றத்தில் ஆஜராகுமாறு அழைப்பாணை  ஐக்கிய தேசிய கட்சியின் பாராளுமன்ற உறுப்பினர் ரஞ்சன் ராமநாயக்கவிற்கு உயர் நீதிமன்றம் அழைப்பாணை ஒன்றை விடுத்துள்ளது", vocabulary)
#
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# vectorizer = TfidfVectorizer(tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
# train_data_features = vectorizer.fit_transform(vocabulary)
# print(vectorizer.get_feature_names())
# # terms = train_data_features.get_feature_names()
# print(vectorizer.vocabulary_)
#
# # print(train_data_features)
# print(vectorizer.transform(["ரஞ்சனுக்கு நீதிமன்றத்தில் ஆஜராகுமாறு அழைப்பாணை  ஐக்கிய தேசிய கட்சியின் பாராளுமன்ற உறுப்பினர் ரஞ்சன் ராமநாயக்கவிற்கு உயர் நீதிமன்றம் அழைப்பாணை ஒன்றை விடுத்துள்ளது"]).toarray())
