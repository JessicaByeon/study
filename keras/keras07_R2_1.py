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
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)  # #x (20,1) x_train (14,1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)  #loss로만 평가하기엔 

y_predict = model.predict(x)

from sklearn.metrics import r2_score  #R2 '결정계수'를 이용 / 정확도 개념과 비슷
r2 = r2_score(y, y_predict) #원래 y값과 y예측값 비교
print('r2 스코어 : ', r2)


# loss :  1.5352953672409058
# r2 스코어 :  0.737962610456472







# import matplotlib.pyplot as plt #맷플롯립 그림 그릴 때 유용
# plt.scatter(x,y)  #scatter : 점을 흩뿌리다/찍다
# plt.plot(x, y_predict, color='red')  #plot : 선을 그리다
# plt.show()