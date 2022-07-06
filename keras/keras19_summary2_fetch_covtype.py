# keras18_gpu_test3 파일의 summary를 확인해보시오.
# summary and time!
# epochs 10

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import pandas as pd

import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

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

#2. 모델구성
model = Sequential()
model.add(Dense(500, activation='linear', input_dim=54))
model.add(Dense(400, activation='sigmoid'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='linear'))
model.add(Dense(7, activation='sigmoid'))

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 500)               27500
_________________________________________________________________
dense_1 (Dense)              (None, 400)               200400
_________________________________________________________________
dense_2 (Dense)              (None, 300)               120300
_________________________________________________________________
dense_3 (Dense)              (None, 300)               90300
_________________________________________________________________
dense_4 (Dense)              (None, 300)               90300
_________________________________________________________________
dense_5 (Dense)              (None, 300)               90300
_________________________________________________________________
dense_6 (Dense)              (None, 300)               90300
_________________________________________________________________
dense_7 (Dense)              (None, 300)               90300
_________________________________________________________________
dense_8 (Dense)              (None, 400)               120400
_________________________________________________________________
dense_9 (Dense)              (None, 7)                 2807
=================================================================
Total params: 922,907
Trainable params: 922,907
Non-trainable params: 0
'''

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)

import time
start_time = time.time() # 현재 시간 출력
hist = model.fit(x_train, y_train, epochs=10, batch_size=500, 
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
print('걸린시간 : ', end_time)
#======================================================================================
model.summary()
# 걸린시간 비교 batch_size=5000
# cpu :  72.5228762626648 
# gpu :  20.271857738494873

# 걸린시간 비교 batch_size=1000
# cpu :  99.73007893562317
# gpu :  34.63421964645386

# 걸린시간 비교 batch_size=500
# cpu :  113.78163313865662
# gpu :  54.99496412277222

#======================================================================================

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

