#1. R2를 음수가 아닌 0.5 이하로 만들것
#2. 데이터 건들지 말것
#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. batch_size=1
#5. 히든레이어의 노드는 10개 이상 100개 이하
#6. train 70%
#7. epoch 100번 이상
#8. loss 지표는 mse, mae


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1) #x (20,1) x_train (14,1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score  #R2 '결정계수'를 이용 / 정확도 개념과 비슷
r2 = r2_score(y, y_predict) #원래 y값과 y예측값 비교
print('r2 스코어 : ', r2)

# loss :  11.87948226928711
# r2 스코어 :  0.44959252311842135


''' 
추가 테스트 결과
#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(10))
model.add(Dense(80))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)  # #x (20,1) x_train (14,1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score  #R2 '결정계수'를 이용 / 정확도 개념과 비슷
r2 = r2_score(y, y_predict) #원래 y값과 y예측값 비교
print('r2 스코어 : ', r2)

# loss :  12.71895694732666
# r2 스코어 :  0.4209019579994252

추가 테스트 결과
#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(10))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(40))
model.add(Dense(70))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)  # #x (20,1) x_train (14,1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score  #R2 '결정계수'를 이용 / 정확도 개념과 비슷
r2 = r2_score(y, y_predict) #원래 y값과 y예측값 비교
print('r2 스코어 : ', r2)

# loss :  13.465373039245605
# r2 스코어 :  0.39893910133947086


'''