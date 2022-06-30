from sympy import evaluate
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np


#1.데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# [실습] 슬라이싱
x_train = x[:12]
y_train = y[:12]
x_test = x[12:14]
y_test = y[12:14]
x_val = x[14:]
y_val = y[14:]

#x_train = np.array(range(1, 11))
#y_train = np.array(range(1, 11))
#x_test = np.array([11,12,13])
#y_test = np.array([11,12,13])
#x_val = np.array([14,15,16])
#y_val = np.array([14,15,16]) #실제 데이터는 1~16이고, 훈련은 1~10, evalutation 11~13, validation 14~16


#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val)) # val_loss(검증로스)는 보통 loss(일반로스)보다 크게 나오는 것이 정상

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)