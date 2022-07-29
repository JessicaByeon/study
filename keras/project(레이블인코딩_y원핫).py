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

x = pd.DataFrame(pad_x1)
print(x)

# 다중분류 -- 원핫인코딩, sparse to cate~ 써줘도 됨
from tensorflow.python.keras.utils.np_utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) # (220, 220)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66)
print(x_train.shape, x_test.shape) # (154, 13) (66, 13)
print(y_train.shape, y_test.shape) # (154, 220) (66, 220)


'''

#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding
model = Sequential()
model.add(Embedding(input_dim=13,output_dim=220,input_length=13)) 
model.add(LSTM(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(220,activation='softmax'))
model.summary() #Total params: 5,847

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=5000)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)
y_predict = model.predict(x_test)

'''






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