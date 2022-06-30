from sympy import evaluate
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1.데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# [실습] train_test_split 으로만 나누기
# 10:3:3 으로 나누기 --- 10개의 데이터는 훈련, 3개 테스트, 3개 검증

x_train,x_test,y_train,y_test = train_test_split(
    x, y, 
    test_size=0.1875,
    shuffle=True, #default가 true이므로 없어도 됨
    random_state=66
    ) #***랜덤난수(랜덤값고정을위해), 난수표 참고, 난수값을 66을 씀

print(x_train.shape, x_test.shape) # (13,) (3,)

#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25) # val_loss(검증로스)는 보통 loss(일반로스)보다 크게 나오는 것이 정상

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)

