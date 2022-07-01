from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y) #y는 데이터 전처리를 할 필요가 없음.
print(x.shape, y.shape) #(442, 10) (442,) Number of Instances: 442, Number of Attributes 10

print(datasets.feature_names)
print(datasets.DESCR)

# [실습]
# R2 0.62 이상

#[실습] validation split 했을 때와 아닐 때의 데이터 손실 비교
#1. test 0.2
#2. validation 0.25


#안했을때
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=72)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(12))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=250, batch_size=40)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  2264.3271484375
# r2 스코어 :  0.6570873599850526


#했을때
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=72)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(12))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=250, batch_size=40, validation_split=0.25)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  2343.607177734375
# r2 스코어 :  0.6450810741559115

# 비교결과 : validation split 사용 시 손실은 늘고, r2 스코어는 줄어듦