from keras.datasets import imdb
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000 # 단어사전의 갯수 빈도수가 높은것부터 10000개를 빼서 데이터 셋으로 지정
)

print(x_train)
print(x_train.shape, x_test.shape) # (25000,) (25000,) 리스트가 8982!, 2246!
print(y_train) # [1 0 0 ... 0 1 0]
print(np.unique(y_train, return_counts=True))
# (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))unique 값 2
print(len(np.unique(y_train))) # 2

print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0])) # <class 'list'> 리스트의 길이는 일정하지 않음 --- pad sequences!
# print(x_train[0].shape) # AttributeError: 'list' object has no attribute 'shape'
print(len(x_train[0])) # 218
print(len(x_train[1])) # 189

# 리스트 길이가 모두 다르기 때문에 padding 으로 길이를 일정하게 맞춰줘야함!

# [확인할것] print(len(max(x_train))) #83??????

# len(i) for i in x_train # 리스트 8982개의 길이 값이 모두 저장됨
print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) # 2494
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) # 238.71364


##### [실습] #####


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

#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')
model = Sequential()
model.add(Embedding(input_dim=46, output_dim=10, input_length=100))
model.add(LSTM(32))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.add(Dense(2,activation='sigmoid'))
model.summary() #Total params: 5,847

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 100, 10)           460
# _________________________________________________________________
# lstm (LSTM)                  (None, 32)                5504
# _________________________________________________________________
# dense (Dense)                (None, 32)                1056
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 33
# _________________________________________________________________
# dense_3 (Dense)              (None, 2)                 4
# =================================================================
# Total params: 8,113
# Trainable params: 8,113
# Non-trainable params: 0
# _________________________________________________________________

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train, y_train, epochs=3, batch_size=5000)

#4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print('acc :', acc)
# y_predict = model.predict(x_test)
# print('predict :', y_predict)

# acc : 0.5
