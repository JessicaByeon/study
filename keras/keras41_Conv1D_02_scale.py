import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM, Conv1D, Flatten


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


#2-1. 모델구성
model = Sequential()
# model.add(SimpleRNN(32, input_shape=(3,1), return_sequences=True))
# model.add(Bidirectional(SimpleRNN(64)))
model.add(Conv1D(32, 2, input_shape=(3,1))) # filter / kenner_size / input_shape 순서로 써줌
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
# model.summary()


# #2-2. 모델구성
# model = Sequential()
# model.add(Bidirectional(SimpleRNN(32, return_sequences=True), input_shape=(3,1)))
# # retrun_sequences 와 input shape 의 위치를 바꿔주니 실행가능
# # Bidirectional 에서는 return_sequences 를 제공하지 않음. RNN 에서만 Bi~ 기능 제공
# model.add(Bidirectional(LSTM(64)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1))
# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=100)

#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = np.array([50,60,70]).reshape(1,3,1) # 현재 (1,3) -> 3차원으로 변경해줘야 함 [[[56], [60], [70]]]
result = model.predict(y_pred)
print('loss :', loss)
print('[50,60,70]의 결과: ', result)


# Conv1D
# loss : 2.489003896713257
# [50,60,70]의 결과:  [[85.24882]]

# Bidirectional
# loss : 7.299648859770969e-05
# [50,60,70]의 결과:  [[70.52212]]

# 기존 결과값
# loss : 0.0005302370409481227
# [50,60,70]의 결과:  [[78.778496]]


