import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN

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
model.add(SimpleRNN(10, input_shape=(3,1))) # 3차원으로 받아서 2차원으로 던져주므로 바로 dense 붙일 수 있음
# model.add(SimpleRNN(32)) # 3차원-> 2차원으로 변경된 상태에서 다시 3차원으로 연결해야하므로 shape이 맞지 않아 바로 RNN으로 연결이 불가능
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
model.summary()


# keras.io -> simplernn layer 검색
# inputs: A 3D tensor, with shape [batch, timesteps, feature]


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


'''
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

# loss : 0.000527280499227345
# [8,9,10]의 결과:  [[10.659381]]

# loss : 0.00015145630459301174
# [8,9,10]의 결과:  [[10.715396]]

# loss : 1.214459007314872e-05
# [8,9,10]의 결과:  [[10.716077]]

# loss : 0.0001026004392770119
# [8,9,10]의 결과:  [[10.618641]]

# loss : 4.457191948858963e-07
# [8,9,10]의 결과:  [[10.773343]]

# loss : 0.0003350088663864881
# [8,9,10]의 결과:  [[10.740458]]
'''