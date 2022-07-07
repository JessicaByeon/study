from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 5)                 10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 10
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 3
=================================================================
Total params: 57
Trainable params: 57
Non-trainable params: 0

# 파라미터 갯수에 b(바이어스) 노드도 추가되어 계산
(1+1) * 5 = 10
(5+1) * 3 = 18
(3+1) * 4 = 16
(4+1) * 2 = 10
(1+1) * 1 = 2

'''

