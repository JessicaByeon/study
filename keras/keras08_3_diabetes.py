from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.9, shuffle=True, random_state=72)

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

# loss :  1930.2774658203125
# r2 스코어 :  0.666016236336831