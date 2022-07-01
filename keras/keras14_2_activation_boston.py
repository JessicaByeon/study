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


# [과제]
# # activation : sigmoid, relu, linear
# metrics 추가
# EarlyStopping 넣고
# 성능비교
# 느낀점 2줄 이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='linear', input_dim=13))
model.add(Dense(9, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, 
                              restore_best_weights=True) 

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) #x 대신 훈련시키지 않은 부분인 x_test로 예측

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #y 대신 y_test
print('r2 스코어 : ', r2)

# loss :  [15.087396621704102, 0.0]
# r2 스코어 :  0.8194918835539146




# 1/ validation 적용
# loss :  27.93036460876465
# r2 스코어 :  0.6658364981018472

# 2/ EarlyStopping 적용
# loss :  19.265213012695312
# r2 스코어 :  0.6678523963782566

# 3/ activation 적용
# loss :  [15.087396621704102, 0.0]
# r2 스코어 :  0.8194918835539146

# 1/,2/,3/을 거치며 점점 손실도 줄고 r2 스코어도 증가하는 것을 발견!
# 최적화가 이루어진 것으로 보인다.