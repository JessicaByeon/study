from pyexpat import model
from hamcrest import starts_with
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
x, y = datasets.data, datasets.target # 한번에 이렇게 쓸 수 있음!

print(x) #8가지 feature
print(y) #보스턴 집값
print(x.shape, y.shape)    #(506, 13) (506,) 데이터 갯수 506, 컬럼 13 / 506개의 스칼라(데이터), 1개의 벡터

print(datasets.feature_names) #사이킷런에서 제공하는 예제 데이터만 가능
 #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B'(흑인) 'LSTAT']
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) # 위 2줄을 이 한줄로 표현가능 fit.transform
x_test = scaler.transform(x_test) # x_train은 fit, transform 모두 실행, x_test는 transform만! fix X!

'''
#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()

# import time
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping =EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, 
                             restore_best_weights=True) 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
                      save_best_only=True,
                      filepath='./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5' # 가장 낮은 지점이 이 경로에 저장, 낮은 값이 나올 때마다 계속적으로 갱신하여 저장
                      )

# start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=20, 
                validation_split=0.2,
                callbacks=[earlyStopping, mcp], # 최저값을 체크해 반환해줌
                verbose=1)
# end_time = time.time()
'''

model = load_model('./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)


# 최저값이 개선되면 다음과 같은 메시지, Epoch 00077: val_loss improved from 24.99981 to 24.95374, saving model to ./_ModelCheckPoint\keras24_ModelCheckPoint.hdf5
# 최저값이 개선되지 않으면 다음과 같은 메시지, Epoch 00087: val_loss did not improve from 19.53281
# Epoch 00087: early stopping
# 4/4 [==============================] - 0s 664us/step - loss: 10.4859
# loss :  10.485852241516113
# r2 스코어 :  0.8745455756521225


