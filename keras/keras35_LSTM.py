# LSTM을 임포트 해온다.

import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
# y = ?

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]])
y = np.array([4,5,6,7,8,9,10])

# x의_shape = (행, 열, 몇개씩 자르는지!) --- 최소 연산 단위, 데이터를 자르는 단위
# RNN 은 3차원의 shape을 가지고 있음
print(x.shape, y.shape) # (7, 3) (7, )
x = x.reshape(7, 3, 1)
print(x.shape) # (7, 3, 1)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(3,1)))
# model.add(SimpleRNN(10, input_length=3, input_dim=1)) # 이렇게 분리해서 쓸 수도 있음
# model.add(SimpleRNN(10, input_dim=1, input_length=3)) # dim 과 length의 순서를 바꿔 이렇게 쓸 수도 있지만 가독성이 떨어짐

model.add(LSTM(32, input_shape=(3,1))) ### SimpleRNN 부분만 LSTM 으로 바꿔주면 간단하게 모델변경!
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape        
#       Param #
# =================================================================
# lstm (LSTM)                  (None, 10)
#       480
# _________________________________________________________________
# dense (Dense)                (None, 5)
#       55
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)
#       6
# =================================================================
# Total params: 541
# Trainable params: 541
# Non-trainable params: 0
# _________________________________________________________________

# LSTM 성능이 좋다 -> 연산량이 많다 -> 기존에 120개였던 param# 가 4배인 480개가 됨.
# param # = unit * (input dim/feature + bias + unit)
# forget gate (망각게이트)
# param # = unit * (input dim/feature + bias + unit)
# [simpleRNN] : 10 -> 10 * (1+1+10)     = 120
# [LSTM]      : 10 -> 4 * 10 * (1+1+10) = 480
#               20 -> 4 * 20 * (1+1+20) = 1760 
# 결론 : LSTM - simpleRNN * 4
# 숫자 4의 의미는 cell state, forget gate, input gate, output gate


# simpleRNN +======================================================

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape        
#       Param #
# =================================================================
# simple_rnn (SimpleRNN)       (None, 10)  일반 dense에 비해(40) 3배나 많음.
#       120
# _________________________________________________________________
# dense (Dense)                (None, 5)
#       55
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)
#       6
# =================================================================
# Total params: 181
# Trainable params: 181
# Non-trainable params: 0
# _________________________________________________________________

# param # = unit * (input dim/feature + bias + unit)
# 10*(1+1+10)=10*12=120

# dnn : unit * (feature(input dim) + bias) 와의 차이 --- rnn unit(output node의 갯수)이 한 번 더 더해지는 이유 : 한 번 더 연산되므로



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # 회귀모델이므로, 딱 떨어지지 않는 결과 값 예측 mse
# from tensorflow.python.keras.callbacks import EarlyStopping

# earlyStopping =EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, 
#                              restore_best_weights=True) 

model.fit(x, y, epochs=500)

#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1,3,1) # 현재 (1,3) -> 3차원으로 변경해줘야 함 [[[8], [9], [10]]]
result = model.predict(y_pred)
print('loss :', loss)
print('[8,9,10]의 결과: ', result)


# loss : 0.00010400601604487747
# [8,9,10]의 결과:  [[10.823878]]
