from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd


#1. 데이터

path = './_data/project_01/' # 경로 = .현재폴더 /하단
data_set = pd.read_csv(path + 'data_project_(220729).csv')

print(data_set)
print(data_set.shape) # (220, 2)

print(data_set.columns) # Index(['분류', '글귀'], dtype='object')
# print(data_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
# print(data_set.describe())

x = data_set['분류']
token = Tokenizer()
token.fit_on_texts(x)
print(token.word_index)
# print(len(token.word_index)) # 244
x1 = token.texts_to_sequences(x)
print(x1)

from keras.preprocessing.sequence import pad_sequences
pad_x1 = pad_sequences(x1, padding='pre', maxlen=13)

print(pad_x1)
print(pad_x1.shape) # (220, 13)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le = le.fit(data_set['글귀'])   # data_set['글귀']을 fit
data_set['글귀'] = le.transform(data_set['글귀'])   # data_set['글귀']에 따라 encoding
y = data_set['글귀']
print(y)
print(y.shape)


col_name = range(220)
print(col_name)

list_df = pd.DataFrame(pad_x1)
print(list_df)

df = pd.concat([list_df,y], axis=1)
print(df) # (220, 14)

# word_size = len(token.word_index)
# print("word_size : ", word_size) # 257 단어사전의 갯수
# print(np.unique(df, return_counts=True))

# 훈련/테스트 데이터 나누기
from sklearn.model_selection import train_test_split
list_df_train, list_df_test, y_train, y_test = train_test_split(
    list_df, y, train_size=0.7, random_state=66)
print(list_df.shape, list_df.shape) # (220, 13) (220, 13)
print(y_train.shape, y_test.shape) # (154,) (66,)










# data_set 은 (978, 2) : 978개의 행과, 2개의 열(분류, 글귀)로 구성

# 1) 데이터

# 1/ x(분류(감정)) 와 y(글귀) 를 분리해서 토크나이징
# 2/ 전체를 한번에 토크나이징한 기준으로... x 와 y 수치화해서 표현해야 같은 단어가 같은 수치를 갖게 됨
# 3/ x는 값이 하나로 구성된 하나의 열이고 유니크 값이 244개...
# 4/ y는 문장들로 구성되어 있으며, 수치화시킬 시 단어들의 리스트로 표시... 
#    행별로 리스트 길이가 다르므로 패딩하여 길이를 일정하게 맞춰줘야함
# 5/ x와 y를 train set, test set 을 분류하여 데이터 준비

# 2) 모델구성
# 1/ 임베딩, LSTM, Dense --- Sequential 모델로...
# 2/ 다중분류 softmax 사용

# 3) 컴파일, 훈련
# 1/ 컴파일 loss=categorical_crossentropy, optimizer='adam', metrics=['acc']
# 2/ 훈련 EarlyStopping



# 원핫인코딩
# import numpy as np
# import pandas as pd
# x = pd.get_dummies(data_set, columns = ['분류'])

# print(x.shape)