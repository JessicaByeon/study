#sklearn.datasets.fetch_california_housing
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(20640, 8) (20640,)

print(datasets.feature_names)
print(datasets.DESCR)


#[실습] validation split 했을 때와 아닐 때의 데이터 손실 비교
#1. test 0.2
#2. validation 0.25


#안했을때
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=20)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  0.7110433578491211
# r2 스코어 :  0.4899495492125632


#했을때
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=20, validation_split=0.25)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  0.6926966309547424
# r2 스코어 :  0.5031103996989461


# 비교결과 : validation split 사용 시 손실은 줄고, r2 스코어는 늘어남