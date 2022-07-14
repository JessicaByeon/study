import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D, Flatten

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
# model.add(LSTM(10, input_shape=(3,1), return_sequences=False))
model.add(Conv1D(10, 2, input_shape=(3,1))) # filter / kenner_size / input_shape 순서로 써줌
model.add(Flatten())
model.add(Dense(3, activation='relu'))
model.add(Dense(1))
# model.summary() # total param# LSTM : 517 // Conv1D : 97


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # 회귀모델이므로, 딱 떨어지지 않는 결과 값 예측 mse
# from tensorflow.python.keras.callbacks import EarlyStopping

# earlyStopping =EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, 
#                              restore_best_weights=True) 

model.fit(x, y, epochs=1000, batch_size=1000)

#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1,3,1) # 현재 (1,3) -> 3차원으로 변경해줘야 함 [[[8], [9], [10]]]
result = model.predict(y_pred)
print('loss :', loss)
print('[8,9,10]의 결과: ', result)


# Conv1D
# loss : 0.046440523117780685
# [8,9,10]의 결과:  [[11.334791]]

# Bidirectional
# loss : 0.10404156148433685
# [8,9,10]의 결과:  [[9.251261]]



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
