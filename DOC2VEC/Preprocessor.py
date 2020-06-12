import codecs
import string
import re
file = open("C:\\Users\\ffayaza\\Documents\\Data\\PROJECT_DATA\\TEST_DATA\\Title_Preprocessed_data\\data_30112018.txt","r", encoding='utf-8')
# C:\\Users\\ffayaza\\Documents\\Data\\PROJECT_DATA\\TEST_DATA\\preprocessed_data\\dec\\data_03122018.txt
# w_file = codecs.open("C:\\Users\\ffayaza\\Documents\\Data\\DOC2VEC\\DOC2VEC_DATA1.txt", encoding='utf8')
f = open("C:\\Users\\ffayaza\\Documents\\Data\\PROJECT_DATA\\TEST_DATA\\Title_final_data\\data_30112018.txt", "w", encoding="utf-8")
for doc in file:
    doc = re.sub(r'[+?.^$()\[!\]=:{}"",'';|]', '', doc)
    if not doc.strip(): continue
    f.write(doc)
f.close()
file.close()

# translator = str.maketrans('', '', string.punctuation)
# string_name = doc.translate(string.punctuation)
# doc = re.sub('\d +', '', doc)
# doc.strip()
# print(doc)
# f.write(doc)