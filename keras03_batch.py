import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])

#2. 모델구성
model = Sequential()
model.add(Dense(36, input_dim=1))
model.add(Dense(36))
model.add(Dense(36))
model.add(Dense(36))
model.add(Dense(36))
model.add(Dense(36))
model.add(Dense(36))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=1050, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('6의 예측값 :', result)

# loss :  0.43812352418899536
# 6의 예측값 : [[5.987179]]


# batch size 조절을 통한 연산가능(batch 작업 _ line 22 참고)
# hyper parameter tunning의 한 가지로 볼 수 있음
# batch size를 줄이게 되면
# 장점
# 1/ 메모리를 적게 차지, 메모리 소모량의 감소
# 2/ 훈련량이 많아지므로 로스값이 줄어듦
# 단점
# 시간이 오래 걸림