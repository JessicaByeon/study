import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, GRU

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


# x의_shape = (행, 열, 몇개씩 자르는지!) --- 최소 연산 단위, 데이터를 자르는 단위
# RNN 은 3차원의 shape을 가지고 있음
print(x.shape, y.shape) # (13, 3) (13, )

x = x.reshape(13, 3, 1)
print(x.shape) # (13, 3, 1)


# 목표값 80

#2. 모델구성
model = Sequential()
model.add(GRU(256, input_shape=(3,1)))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1))
# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500)

#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = np.array([50,60,70]).reshape(1,3,1) # 현재 (1,3) -> 3차원으로 변경해줘야 함 [[[56], [60], [70]]]
result = model.predict(y_pred)
print('loss :', loss)
print('[50,60,70]의 결과: ', result)

# loss : 7.84667136031203e-05
# [50,60,70]의 결과:  [[77.481865]]