from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x) #8가지 feature
print(y) #보스턴 집값
print(x.shape, y.shape)    #(506, 13) (506,) 데이터 갯수 506, 컬럼 13 / 506개의 스칼라(데이터), 1개의 벡터

print(datasets.feature_names) #사이킷런에서 제공하는 예제 데이터만 가능
 #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B'(흑인) 'LSTAT']
print(datasets.DESCR)



#[실습] validation split 했을 때와 아닐 때의 데이터 손실 비교
#1. test 0.2
#2. validation 0.25


#안했을 때
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1, validation_split=0.25)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) #x 대신 훈련시키지 않은 부분인 x_test로 예측

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #y 대신 y_test
print('r2 스코어 : ', r2)

# loss :  20.824501037597656
# r2 스코어 :  0.7508522391551226



#했을 때
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1, validation_split=0.25)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) #x 대신 훈련시키지 않은 부분인 x_test로 예측

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #y 대신 y_test
print('r2 스코어 : ', r2)

# loss :  23.131698608398438
# r2 스코어 :  0.7232485223337912

# 비교결과 : validation split 사용 시 손실은 늘고, r2 스코어는 줄어듦