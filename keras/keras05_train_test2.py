import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 넘파이 리스트의 슬라이싱, 7:3으로 잘라라
x_train = x[0:7] #[:7] 처음부터 7까지
x_test = x[7:10] #[7:] 7부터 끝까지  *** [:] --> 처음부터 끝까지
y_train = x[0:7]
y_test = x[7:10]

print(x_train) #[1 2 3 4 5 6 7]
print(x_test) #[ 8  9 10]
print(y_train) #[1 2 3 4 5 6 7]
print(y_test) #[ 8  9 10]

# 윗부분에서 훈련 셋과 테스트 셋의 데이터 슬라이싱의 오류!
# 뒷부분을 통째로 뺄 경우 데이터가 한쪽으로 치우쳐 뒷부분의 특성이 무시된 훈련이 될 수 있으므로
# 랜덤하게 shuffle 하게 30% 뺀 데이터로 훈련



# x_train = np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10])
# y_train = np.array([1,2,3,4,5,6,7])
# y_test = np.array([8,9,10])

'''
#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1)) #hidden layer가 하나라도 딥러닝

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
result = model.predict([11])
print('[11]의 예측값 :', result)
'''