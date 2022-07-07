from pyexpat import model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sqlalchemy import false
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
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


#[실습] 아래를 완성할 것
#1. train 0.7
#2. R2 0.8 이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # 수치로 변환해주는 걸 x_train에 집어넣자.
x_test = scaler.transform(x_test) 


#2. 모델구성
# model = Sequential()
# model.add(Dense(64, input_dim=13))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))
# model.summary()

# model.save("./_save/keras23_1_save_model.h5")

model = load_model("./_save/keras23_3_save_model.h5")
# [중요] 이미 저장된 상태에서 모델 아래쪽에 위치시키고 컴파일, 훈련을 적용시켜주면
# 저장된 파일에서 새로운 가중치(랜덤)로 값 갱신


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping =EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, 
                             restore_best_weights=True) 

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=20, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음

end_time = time.time() - start_time

# model.save("./_save/keras23_3_save_model.h5")


# model = load_model("./_save/keras23_3_save_model.h5")
# [중요] fit 다음에 save하면 모델과 가중치까지 모두 저장됨!


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  10.323189735412598
# r2 스코어 :  0.8764916599963213
