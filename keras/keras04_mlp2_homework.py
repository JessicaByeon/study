import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], #2개의 feature를 가짐
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
             [9,8,7,6,5,4,3,2,1,0]])
y = np.array([11,12,13,14,15,16,17,18,19,20])

print(x.shape) #(3,10) -> 이 행렬을 (10, 3) shape 으로 변경하려고 함
print(y.shape) #(10,) -> (10,1)

x = x.T
# x = x.transpose() 해당 2가지 방법은 행과 열을 바꿔주지만
# x = x.reshape(10,3) reshape는 순서를 그대로 유지해줌
print(x)
print(x.shape) # (10.3)


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3)) # (10,3)이므로 dim 3 : dim_열/컬럼/특성/feature의 갯수 (열우선)
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[10, 1.4, 0]]) #괄호
print('[10, 1.4, 0]의 예측값 : ',  result)

# loss :  0.003355496795848012
# [10, 1.4, 0]의 예측값 :  [[19.953024]]