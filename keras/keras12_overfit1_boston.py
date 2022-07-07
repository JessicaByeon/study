<<<<<<< HEAD
# 기존 boston housing 이용하여 valiation split / hist.shistory / matplotlib 적용

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sqlalchemy import false
from tensorflow.python.keras.models import Sequential
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

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# start_time = time.time() #현재 시간 출력 1656032967.2581124
# print(start_time) #1656032967.2581124

hist = model.fit(x_train, y_train, epochs=500, batch_size=20, 
                validation_split=0.2,
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음

# end_time = time.time() - start_time #걸린 시간


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print('------------------------------')
print(hist) # <tensorflow.python.keras.callbacks.History object at 0x00000219A7310F40>
print('------------------------------')
print(hist.history) 
print('------------------------------')
print(hist.history['loss']) #키밸류 상의 loss는 이름이기 때문에 ''를 넣어줌
print('------------------------------')
print(hist.history['val_loss']) #키밸류 상의 val_loss는 이름이기 때문에 ''를 넣어줌

# print("걸린시간 : ", end_time)


# 이 값을 이용해 그래프를 그려보자!

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # 연속된 데이터는 엑스 빼고 와이만 써주면 됨. 순차적으로 진행.
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() # 모눈종이 형태로 볼 수 있도록 함
plt.title('이결바보')
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right') # 라벨값이 원하는 위치에 명시됨
plt.legend()
plt.show()



# {'loss': [1521.7059326171875, 130.54188537597656, 91.58596801757812, 82.65372467041016, 
# 72.5574951171875, 70.0708236694336, 66.46566009521484, 70.04518127441406, 63.4539794921875, 63.71456527709961, 61.172828674316406], 'val_loss': [158.5045623779297, 101.32865905761719, 85.67953491210938, 82.06331634521484, 70.68658447265625, 87.28343200683594, 101.1255874633789, 73.15115356445312, 63.8593635559082, 113.73959350585938, 72.89293670654297]}
# 딕셔너리{} 키밸류 형태로 loss와 val_loss를 반환해줌

# val 없이 반환하면 로스만 반환
# val 적용하여 반환하면 로스, val_loss 두가지 반환



# 히스토리 안에 반환값 - loss. val_loss
=======
# 기존 boston housing 이용하여 valiation split / hist.shistory / matplotlib 적용

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sqlalchemy import false
from tensorflow.python.keras.models import Sequential
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

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# start_time = time.time() #현재 시간 출력 1656032967.2581124
# print(start_time) #1656032967.2581124

hist = model.fit(x_train, y_train, epochs=500, batch_size=20, 
                validation_split=0.2,
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음

# end_time = time.time() - start_time #걸린 시간


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print('------------------------------')
print(hist) # <tensorflow.python.keras.callbacks.History object at 0x00000219A7310F40>
print('------------------------------')
print(hist.history) 
print('------------------------------')
print(hist.history['loss']) #키밸류 상의 loss는 이름이기 때문에 ''를 넣어줌
print('------------------------------')
print(hist.history['val_loss']) #키밸류 상의 val_loss는 이름이기 때문에 ''를 넣어줌

# print("걸린시간 : ", end_time)


# 이 값을 이용해 그래프를 그려보자!

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # 연속된 데이터는 엑스 빼고 와이만 써주면 됨. 순차적으로 진행.
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() # 모눈종이 형태로 볼 수 있도록 함
plt.title('이결바보')
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right') # 라벨값이 원하는 위치에 명시됨
plt.legend()
plt.show()



# {'loss': [1521.7059326171875, 130.54188537597656, 91.58596801757812, 82.65372467041016, 
# 72.5574951171875, 70.0708236694336, 66.46566009521484, 70.04518127441406, 63.4539794921875, 63.71456527709961, 61.172828674316406], 'val_loss': [158.5045623779297, 101.32865905761719, 85.67953491210938, 82.06331634521484, 70.68658447265625, 87.28343200683594, 101.1255874633789, 73.15115356445312, 63.8593635559082, 113.73959350585938, 72.89293670654297]}
# 딕셔너리{} 키밸류 형태로 loss와 val_loss를 반환해줌

# val 없이 반환하면 로스만 반환
# val 적용하여 반환하면 로스, val_loss 두가지 반환



# 히스토리 안에 반환값 - loss. val_loss
>>>>>>> 425a270641f2121145cc0df2169ab516031dcf7e
# 로스에서 최저값을 찾을 수 있음, 그 지점이 최적의 Weight