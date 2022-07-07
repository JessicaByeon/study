# 함수형 모델구성으로 변경

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import pandas as pd

import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) :
    print('쥐피유 돈다')
    aaa = 'gpu'
else:
    print('쥐피유 안도라')
    aaa = 'cpu'

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) 
# [1 2 3 4 5 6 7] dim 7 softmax (1797,54)으로 원핫인코딩 평가지표 accuracy
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
# print(datasets.DESCR)
# print(datasets.feature_names)

# pandas의 get_dummies
y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # 수치로 변환해주는 걸 x_train에 집어넣자.
x_test = scaler.transform(x_test) 
# # print(np.min(x_train))
# # print(np.max(x_train))
# # print(np.min(x_test))
# print(np.max(x_test))

#2. 모델구성
# model = Sequential()
# model.add(Dense(500, activation='linear', input_dim=54))
# model.add(Dense(400, activation='sigmoid'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(400, activation='linear'))
# model.add(Dense(7, activation='sigmoid'))

input1 = Input(shape=(54,)) # 먼저 input layer를 명시해줌
dense1 = Dense(500, activation='linear')(input1)
dense2 = Dense(400, activation='sigmoid')(dense1)
dense3 = Dense(300, activation='relu')(dense2)
dense4 = Dense(300, activation='relu')(dense3)
dense5 = Dense(300, activation='relu')(dense4)
dense6 = Dense(300, activation='relu')(dense5)
dense7 = Dense(300, activation='relu')(dense6)
dense8 = Dense(300, activation='relu')(dense7)
dense9 = Dense(400, activation='linear')(dense8)
output1 = Dense(7, activation='sigmoid')(dense9)
model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)

import time
start_time = time.time() # 현재 시간 출력
hist = model.fit(x_train, y_train, epochs=10, batch_size=100, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1)

#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)

# 아래와 같이 표기도 가능!
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('accuracy : ', result[1])

end_time = time.time() - start_time # 걸린 시간
print(aaa, '걸린시간 : ', end_time)

# 걸린시간 비교
# cpu 걸린시간 :  191.34082531929016
# gpu 걸린시간 :  210.9643955230713

'''
# print("============= y_test[:5] ==============")
# print(y_test[:5])
# print("============= y_pred ==============")
# y_predict = model.predict(x_test[:5])
# print(y_predict)
print("============= y_pred ==============")

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
print(y_predict)
y_test = tf.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)


# loss :  0.6646304726600647
# accuracy :  0.7235957980155945
# ============= y_pred ==============
# tf.Tensor([1 1 0 ... 2 1 1], shape=(116203,), dtype=int64)
# tf.Tensor([1 1 0 ... 5 1 1], shape=(116203,), dtype=int64)       
# acc 스코어 :  0.7235957763568927
'''



#=============================================================================
# loss :  1.019610047340393
# accuracy :  0.5463628172874451
# cpu 걸린시간 :  194.00481414794922
# gpu 걸린시간 :  220.12128043174744
#=============================================================================
# MinMaxScaler
# loss :  0.23077496886253357
# accuracy :  0.9059318900108337
# cpu 걸린시간 :  194.88401222229004
#=============================================================================
# StandardScaler
# loss :  0.21361994743347168
# accuracy :  0.9142448902130127
# cpu 걸린시간 :  198.14081501960754
#=============================================================================
# MaxAbsScaler
# loss :  0.2502190172672272
# accuracy :  0.8973434567451477
# cpu 걸린시간 :  205.43295907974243
#=============================================================================
# RobustScaler
# loss :  0.19143211841583252
# accuracy :  0.9241930246353149
# cpu 걸린시간 :  206.75027465820312

# 함수형 모델 =================================================================
# RobustScaler
# loss :  0.19143211841583252
# accuracy :  0.9241930246353149
# cpu 걸린시간 :  195.82034420967102 