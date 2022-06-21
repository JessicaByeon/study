from tkinter import Y
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
# print(range(10)) # 0~9 까지(10 이전까지를 나타냄)
# for i in range(10): # FOR문 (조건 IF, 반복 FOR)
#     print(i)
print(x.shape) #(3,10)
x = np.transpose(x)
print(x.shape) #(10,3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]])
y = np.transpose(y)
print(y.shape) #(10,2)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(4))
model.add(Dense(13))
model.add(Dense(14))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[9, 30, 210]])
print('[9, 30, 210]의 예측값 : ',  result) #예상 y값 [[10, 1.9]]

# loss :  3.567711530649831e-07
# [9, 30, 210]의 예측값 :  [[9.999853  1.8995806]]