from keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2 #num_words 단어사전의 갯수
)

print(x_train)
print(x_train.shape, x_test.shape) # (8982,) (2246,) 리스트가 8982!, 2246!
print(y_train) # [ 3  4  3 ... 25  3 25]
print(np.unique(y_train, return_counts=True))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], dtype=int64) unique 값 46 (뉴스 카테고리)
print(len(np.unique(y_train))) # 46

print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0])) # <class 'list'> 리스트의 길이는 일정하지 않음 --- pad sequences!
# print(x_train[0].shape) # AttributeError: 'list' object has no attribute 'shape'
print(len(x_train[0])) # 87
print(len(x_train[1])) # 56

# 리스트 길이가 모두 다르기 때문에 padding 으로 길이를 일정하게 맞춰줘야함!

# [확인할것] print(len(max(x_train))) #83??????

# len(i) for i in x_train # 리스트 8982개의 길이 값이 모두 저장됨
print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) # 2376
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) # 145.5398574927633

# 전처리
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
                        # (8982,) -> (8982, 100)
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

# 다중분류 -- 원핫인코딩, sparse to cate~ 써줘도 됨
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) # (8982, 100) (8982, 46)
print(x_test.shape, y_test.shape)   # (2246, 100) (2246, 46)

# https://wikidocs.net/22933


#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding
model = Sequential()
model.add(Embedding(input_dim=46,output_dim=10,input_length=100)) 
model.add(LSTM(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(46,activation='softmax'))
model.summary() #Total params: 5,847

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=5000)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :',loss)
y_predict = model.predict(x_test)

print('predict :', y_predict[-1])

print('predict :', np.round(y_predict[-1],0))

# loss : [3.6291472911834717, 0.21104185283184052]