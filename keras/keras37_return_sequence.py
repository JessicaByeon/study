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


# reshape 후 이제 3차원!
# 이 3차원을 LSTM에 넣어줄 준비!


#2. 모델구성
model = Sequential()
model.add(LSTM(10, return_sequences=True, input_shape=(3,1))) 
# 3차원 shape을 LSTM에 넣어주면 2차원으로! 여기까지 실행하면 2차원으로 출력되는데
# 아래 LSTM에 3차원을 넣어줘야해서 충돌이 생김(2차원과 3차원), 그러므로 차원을 3차원으로 넣어주기위해 return_sequences를 사용!
# return_sequences 를 넣으면 1차원이 늘어나 3차원으로 넣어줄 수 있음 출력값이 (N, 3, 1) -> (N, 3, 10) Filter 부분만 변경

model.add(LSTM(5, return_sequences=False)) # ValueError: Input 0 of layer lstm_1 is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 10)
model.add(Dense(1))
# model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape        
#       Param #
# =================================================================
# lstm (LSTM)                  (None, 3, 10)       
#       480
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 5)
#       320
# _________________________________________________________________
# dense (Dense)                (None, 1)
#       6
# =================================================================
# Total params: 806
# Trainable params: 806
# Non-trainable params: 0
# _________________________________________________________________

'''
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
'''