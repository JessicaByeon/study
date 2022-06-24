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



#[실습] 아래를 완성할 것
#1. train 0.7
#2. R2 0.8 이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

import time
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time() #현재 시간 출력 1656032967.2581124
print(start_time) #1656032967.2581124
model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=0) #verbose=0 일때는 훈련과정을 보여주지 않음
end_time = time.time() - start_time #걸린 시간

print("걸린시간 : ", end_time)

"""
verbose 0 걸린시간 :  9.972111701965332 / 출력없다.
verbose 1 걸린시간 :  12.227246284484863 / 잔소리 많다.
verbose 2 걸린시간 :  10.22326135635376 / 프로그래스바 없다.
verbose 3, 4, 5... 걸린시간 :  10.08649206161499 / epoch만 나온다.


"""