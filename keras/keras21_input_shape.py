import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], #2개의 feature를 가짐
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
             [9,8,7,6,5,4,3,2,1,0]])
y = np.array([11,12,13,14,15,16,17,18,19,20])

print(x.shape, y.shape) #(3,10) (10,)
x = x.T
# x = x.transpose() 해당 2가지 방법은 행과 열을 바꿔주지만
# x = x.reshape(10,3) reshape는 순서를 그대로 유지해줌
print(x.shape) # (10.3)

#2. 모델구성
model = Sequential()
# model.add(Dense(10, input_dim=3)) # (100,3) -> (none, 3) / csv만 가능, 그 이상의 차원은 input_shape 로...
model.add(Dense(10, input_shape=(3,)))
# shape (3,1) 3개짜리 컬럼이 들어갈 경우 input_shape 에 3개
# 열의 갯수가 3으로 들어감, 행 제외 나머지 부분들
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))
model.summary()

# Model: "sequential"
# _____________________________none = 행의 갯수____________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 10)                40
# _________________________________________________________________
# dense_1 (Dense)              (None, 5)                 55
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 18
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 4
# =================================================================
# Total params: 117
# Trainable params: 117
# Non-trainable params: 0