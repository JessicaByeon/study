# layer 에서 dropout 을 import 해줌
# mcp 저장하지 않고 실행

from pyexpat import model
from hamcrest import starts_with
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sqlalchemy import false
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
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

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dropout(0.3)) # 30% 만큼 제외
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
# model.summary()

# input1 = Input(shape=(3,))
# dense1 = Dense(10)(input1)
# dense2 = Dense(5, activation='relu')(dense1)
# drop1 = Dropout(0.2)(dense2)
# dense3 = Dense(3, activation='sigmoid')(drop1)
# output1 = Dense(1)(dense3)
# model = Model(inputs=input1, outputs=output)

# 평가 예측에는 전체 노드에 모두 적용 (dropout 적용하지 않음)



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
                callbacks=[earlyStopping], # 최저값을 체크해 반환해줌
                verbose=1)
# end_time = time.time()


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)


# 최저값이 개선되면 다음과 같은 메시지, Epoch 00123: val_loss improved from 19.88012 to 19.75124, saving model to ./_ModelCheckPoint\keras24_ModelCheckPoint.hdf5
# 최저값이 개선되지 않으면 다음과 같은 메시지, Epoch 00134: val_loss did not improve from 19.53281
# Epoch 00134: early stopping
# 4/4 [==============================] - 0s 332us/step - loss: 9.1255
# loss :  9.125495910644531
# r2 스코어 :  0.8908210680543753


# dropout 사용 결과값
# loss :  12.311768531799316
# r2 스코어 :  0.8526999674290076