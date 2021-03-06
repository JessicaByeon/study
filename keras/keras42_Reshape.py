from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout # 이미지 작업은 2D
from tensorflow.python.keras.layers import Conv1D, LSTM, Reshape

from keras.datasets import mnist
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# [실습] acc 0.98 이상
# 원핫인코딩(to_cat, get dummies)

#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = StandardScaler()
# scaler.fit(x_train)
# scaler.transform(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape) # (60000, 28, 28, 1)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
#  array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))

# reshape 할 때 모든 개체를 곱한 값은 동일해야한다.
# 모양은 바꿀 수 있다. 다만 데이터 순서만 바뀌지 않으면 됨

# pandas의 get_dummies
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), 
                 padding='same', input_shape=(28, 28, 1))) # (batch_size, rows, columns, channels)   # 출력 : (N, 28, 28, 64)
model.add(MaxPooling2D())                  # (N, 14, 14, 64)
model.add(Conv2D(32, (3,3)))               # (N, 12, 12, 32)
model.add(Conv2D(7, (3,3)))                # (N, 10, 10, 7)
model.add(Flatten())                       # (N, 700) 10*10*7
model.add(Dense(100, activation='relu'))   # (N, 100)
model.add(Reshape(target_shape=(100,1)))   # (N, 100, 1)
# 데이터가 아닌 레이어 상에서 reshape! 늘릴 수 있고 줄일(flatten 대신 사용가능) 수 있음
# 순서와 내용은 바뀌지 않고 모양만 바꿔줌, 연산량 없음
model.add(Conv1D(10, 3))                   # (N, 98, 10)
# Conv1D가 받아들이는 차원이 3차원! 출력 3차원
model.add(LSTM(16))                        # (N, 16)
# LSTM 3차원으로 받아 2차원으로 출력
model.add(Dense(32, activation='relu'))    # (N, 32)
model.add(Dense(10, activation='softmax')) # (N, 10)
model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 64)        640
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 12, 12, 32)        18464
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 10, 10, 7)         2023
# _________________________________________________________________
# flatten (Flatten)            (None, 700)               0
# _________________________________________________________________
# dense (Dense)                (None, 100)               70100
# _________________________________________________________________
# reshape (Reshape)            (None, 100, 1)            0
# _________________________________________________________________
# conv1d (Conv1D)              (None, 98, 10)            40
# _________________________________________________________________
# lstm (LSTM)                  (None, 16)                1728
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                544
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                330
# =================================================================
# Total params: 93,869
# Trainable params: 93,869
# Non-trainable params: 0
# _________________________________________________________________

# #2. 모델구성 또 다른 예
# model = Sequential()
# model.add(Conv2D(filters=64, kernel_size=(3,3), 
#                  padding='same', input_shape=(28, 28, 1))) # (batch_size, rows, columns, channels)   # 출력 : (N, 28, 28, 64)
# model.add(MaxPooling2D())                  # (N, 14, 14, 64)
# model.add(Conv2D(32, (3,3)))               # (N, 12, 12, 32)
# model.add(Conv2D(7, (3,3)))                # (N, 10, 10, 7)
# model.add(Reshape(target_shape=(25,28)))   # (N, 100, 7)
# # model.add(Reshape(target_shape=(100,7)))   # (N, 100, 7)
# model.add(Flatten())                       # (N, 700) 10*10*7
# model.add(Dense(100, activation='relu'))   # (N, 100)
# # model.add(Reshape(target_shape=(100,1)))   # (N, 100, 1)
# # model.add(Conv1D(10, 3))                   # (N, 98, 10)
# # model.add(LSTM(16))                        # (N, 16)
# # model.add(Dense(32, activation='relu'))    # (N, 32)
# # model.add(Dense(10, activation='softmax')) # (N, 10)
# model.summary()


'''
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
# print(date) # 2022-07-07 17:21:37.577295 수치형 데이터
date = date.strftime("%m%d_%H%M")
print(date) # 0707_1723 자료형 데이터(문자형)

# 파일명을 계속적으로 수정하지 않고 고정시켜주기 위해
filepath = './_ModelCheckPoint/k28/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # d4 네자리까지, .4f 소수넷째자리까지

earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
                      save_best_only=True,
                      filepath= "".join([filepath, 'k28_', date, '_', filename] # .join안에 있는 모든 문자열을 합치겠다.
                      ))

# start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=200, 
                validation_split=0.2,
                callbacks=[earlyStopping, mcp], # 최저값을 체크해 반환해줌
                verbose=1)
# end_time = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, accuracy_score
y_predict = np.argmax(y_predict, axis= 1)
y_predict = to_categorical(y_predict)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)

# loss :  [0.08913582563400269, 0.9805999994277954]
# acc스코어 :  0.9806
'''