from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D # 이미지 작업은 2D
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# [실습] acc 0.98 이상
# 기준값 0.68 / 원핫인코딩(to_cat, get dummies)

#1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape) # (60000, 28, 28, 1)
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
#  array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))

# reshape 할 때 모든 개체를 곱한 값은 동일해야한다.
# 모양은 바꿀 수 있다. 다만 데이터 순서만 바뀌지 않으면 됨

# pandas의 get_dummies
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), # 10 다음 레이어로 전달해주는 아웃풋 노드의 갯수, kernel size 이미지를 자르는 규격
                 padding='same', # 원래 shape를 그대로 유지하여 다음 레이어로 보내주고 싶을 때 주로 사용!
                 input_shape=(28, 28, 1))) # (batch_size, rows, columns, channels)   # 출력 : (N, 28, 28, 64)
model.add(MaxPooling2D()) # (N, 14, 14, 64)
model.add(Conv2D(32, (2,2), 
                 padding='valid', # 디폴트
                 activation='relu')) # filter = 7, kernel size = (2,2) # 출력 : (N, 13, 13, 32)

model.add(Flatten()) # (N, 175) 5*5*7
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 

hist = model.fit(x_train, y_train, epochs=10, batch_size=100, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  [2.3466641902923584, 0.3458999991416931]
# r2 스코어 :  -0.002594790187702556