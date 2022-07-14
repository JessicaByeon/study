# 기존 DNN 모델과 LSTM 모델의 성능비교 (1번~12번)
# 차원 비교 후 reshape 이용

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sqlalchemy import false
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM

datasets = load_boston()
x = datasets.data
y = datasets['target']

# print(np.min(x)) # 0.0
# print(np.max(x)) # 711.0 --- 711을 최대값인 1로 잡고 계산
# x = (x - np.min(x)) / (np.max(x) - np.min(x))
# print(x[:10])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train) # 수치로 변환해주는 걸 x_train에 집어넣자.
x_train = scaler.fit_transform(x_train) # 위 2줄을 이 한줄로 표현가능 fit.transform
x_test = scaler.transform(x_test) # x_train은 fit, transform 모두 실행, x_test는 transform만! fix X!
# print(np.min(x_train)) # 0.0
# print(np.max(x_train)) # 1.0000000000000002
# print(np.min(x_test)) # -0.06141956477526944
# print(np.max(x_test)) # 1.1478180091225068

print(x_train.shape, x_test.shape) # (354, 13) (152, 13)
print(y_train.shape, y_test.shape) # (354,) (152,)
x_train = x_train.reshape(354, 13, 1)
x_test = x_test.reshape(152, 13, 1)
print(x_train.shape, x_test.shape) # (354, 13, 1) (152, 13, 1)


x_train = x_train.reshape(404, 13, 1, 1)
x_test = x_test.reshape(102, 13, 1, 1)
print(np.unique(y_train, return_counts=True))


#2. 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(13,1), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
# model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=100, 
          validation_split=0.2,
          verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# LSTM
# loss :  53.72352981567383
# r2 스코어 :  0.3497286573185798



#=============================================================================
# loss :  21.426353454589844
# r2 스코어 :  0.7406547626963109
#=============================================================================
# MinMaxScaler
# loss :  17.020431518554688
# r2 스코어 :  0.793984149564475
#=============================================================================
# StandardScaler
# loss :  16.821285247802734
# r2 스코어 :  0.7963946205369385
#=============================================================================
# MaxAbsScaler
# loss :  17.01386070251465
# r2 스코어 :  0.7940637072715339
#=============================================================================
# RobustScaler
# loss :  17.2054500579834
# r2 스코어 :  0.7917446902703928

# 스케일링을 했을 때 더 결과값이 좋음. 
# 해당 모델에서는 StandardScaler가 가장 좋은 결과값을 가짐.

