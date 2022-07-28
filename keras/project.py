from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd


#1. 데이터

path = './_data/project_01/' # 경로 = .현재폴더 /하단
data_set = pd.read_csv(path + 'data_project.csv')

print(data_set)
print(data_set.shape) # (978, 2)

print(data_set.columns) # Index(['분류', '글귀'], dtype='object')
print(data_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
print(data_set.describe())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 978 entries, 0 to 977
# Data columns (total 2 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   분류      978 non-null    object
#  1   글귀      978 non-null    object
# dtypes: object(2)
# memory usage: 15.4+ KB
# None
#          분류                                                 글귀
# count   978                                                978
# unique  244                                                222
# top      노력  모든 슬픔은 그것을 이야기로 만들거나 그것에 대해 이야기 할 수 있다면 견뎌질 수 ...
# freq     53                                                 14

x = data_set['분류']
token = Tokenizer()
token.fit_on_texts(x)
print(token.word_index)
# print(len(token.word_index)) # 244
x = token.word_index
print(x)

# y = data_set['글귀']
# y = token.texts_to_sequences(y)
# print(y)

y = data_set['글귀']
token = Tokenizer()
token.fit_on_texts(y)
word_index = token.word_index

y = token.texts_to_sequences(y)

print(word_index)
print(y)















'''
from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=6) 
print(pad_x)
print(pad_x.shape)



# 원핫인코딩
# import numpy as np
# import pandas as pd
# x = pd.get_dummies(data_set, columns = ['분류'])

# print(x.shape) # (978, 245)


# 훈련/테스트 데이터 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66)

print(x_train.shape, x_test.shape)
# (336264, 20, 13) (84067, 20, 13)



import csv
x = csv.reader(data_set)
print(list(x))
'''
'''
x = data_set['분류']
y = data_set['글귀']
token = Tokenizer()
token.fit_on_texts(x)
print(token.word_index)
print(len(token.word_index)) # 256

x = token.word_index
print(x)

token.fit_on_texts(y)
print(token.word_index)
print(len(token.word_index)) # 1883

y = token.word_index
print(y)


x1 = token.texts_to_sequences(x)
print(x1)
y1 = token.texts_to_sequences(y)
print(y1)

print(x)



# print(x, x.shape)
# print(y, y.shape)
# print(type(x), type(y))

# print("분류최대길이 : ", max(len(i) for i in x)) # 38
# print("글귀최대길이 : ", max(len(i) for i in y)) # 134


# from keras.preprocessing.sequence import pad_sequences
# pad_y = pad_sequences(y, padding='pre', maxlen=13) 
# print(pad_y)
# print(pad_y.shape)


'''