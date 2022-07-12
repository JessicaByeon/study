<<<<<<< HEAD
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import pandas as pd

import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) 
# [1 2 3 4 5 6 7] dim 7 softmax (1797,54)으로 원핫인코딩 평가지표 accuracy
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
# print(datasets.DESCR)
# print(datasets.feature_names)

# pandas의 get_dummies
y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # 수치로 변환해주는 걸 x_train에 집어넣자.
x_test = scaler.transform(x_test) 
# # print(np.min(x_train))
# # print(np.max(x_train))
# # print(np.min(x_test))
# print(np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(500, activation='linear', input_dim=54))
model.add(Dropout(0.3)) # 30% 만큼 제외
model.add(Dense(400, activation='sigmoid'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(400, activation='linear'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(7, activation='sigmoid'))

# input1 = Input(shape=(54,)) # 먼저 input layer를 명시해줌
# dense1 = Dense(500, activation='linear')(input1)
# dense2 = Dense(400, activation='sigmoid')(dense1)
# dense3 = Dense(300, activation='relu')(dense2)
# dense4 = Dense(300, activation='relu')(dense3)
# dense5 = Dense(300, activation='relu')(dense4)
# dense6 = Dense(300, activation='relu')(dense5)
# dense7 = Dense(300, activation='relu')(dense6)
# dense8 = Dense(300, activation='relu')(dense7)
# dense9 = Dense(400, activation='linear')(dense8)
# output1 = Dense(7, activation='sigmoid')(dense9)
# model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
# print(date)
date = date.strftime("%m%d_%H%M")
# print(date) 

 # 파일명을 계속적으로 수정하지 않고 고정시켜주기 위해
filepath = './_ModelCheckPoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # d4 네자리까지, .4f 소수넷째자리까지

earlyStopping =EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, 
                             restore_best_weights=True) 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
                      save_best_only=True,
                      filepath= "".join([filepath, 'k24_', date, '_', filename] # .join안에 있는 모든 문자열을 합치겠다.
                      ))

hist = model.fit(x_train, y_train, epochs=10, batch_size=100, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음


#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)

# 아래와 같이 표기도 가능!
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('accuracy : ', result[1])

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
print(y_predict)
y_test = tf.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)


# loss :  0.19143211841583252
# accuracy :  0.9241930246353149
# tf.Tensor([1 0 1 ... 5 1 1], shape=(116203,), dtype=int64)
# tf.Tensor([1 1 0 ... 5 1 1], shape=(116203,), dtype=int64)
# acc 스코어 :  0.9238315706134953

# dropout 사용 결과값
# loss :  0.3311381936073303
# accuracy :  0.8697279691696167
# tf.Tensor([1 0 1 ... 5 1 1], shape=(116203,), dtype=int64)
# tf.Tensor([1 1 0 ... 5 1 1], shape=(116203,), dtype=int64)
=======
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import pandas as pd

import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) 
# [1 2 3 4 5 6 7] dim 7 softmax (1797,54)으로 원핫인코딩 평가지표 accuracy
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
# print(datasets.DESCR)
# print(datasets.feature_names)

# pandas의 get_dummies
y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # 수치로 변환해주는 걸 x_train에 집어넣자.
x_test = scaler.transform(x_test) 
# # print(np.min(x_train))
# # print(np.max(x_train))
# # print(np.min(x_test))
# print(np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(500, activation='linear', input_dim=54))
model.add(Dropout(0.3)) # 30% 만큼 제외
model.add(Dense(400, activation='sigmoid'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(400, activation='linear'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(7, activation='sigmoid'))

# input1 = Input(shape=(54,)) # 먼저 input layer를 명시해줌
# dense1 = Dense(500, activation='linear')(input1)
# dense2 = Dense(400, activation='sigmoid')(dense1)
# dense3 = Dense(300, activation='relu')(dense2)
# dense4 = Dense(300, activation='relu')(dense3)
# dense5 = Dense(300, activation='relu')(dense4)
# dense6 = Dense(300, activation='relu')(dense5)
# dense7 = Dense(300, activation='relu')(dense6)
# dense8 = Dense(300, activation='relu')(dense7)
# dense9 = Dense(400, activation='linear')(dense8)
# output1 = Dense(7, activation='sigmoid')(dense9)
# model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
# print(date)
date = date.strftime("%m%d_%H%M")
# print(date) 

 # 파일명을 계속적으로 수정하지 않고 고정시켜주기 위해
filepath = './_ModelCheckPoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # d4 네자리까지, .4f 소수넷째자리까지

earlyStopping =EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, 
                             restore_best_weights=True) 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
                      save_best_only=True,
                      filepath= "".join([filepath, 'k24_', date, '_', filename] # .join안에 있는 모든 문자열을 합치겠다.
                      ))

hist = model.fit(x_train, y_train, epochs=10, batch_size=100, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음


#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)

# 아래와 같이 표기도 가능!
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('accuracy : ', result[1])

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
print(y_predict)
y_test = tf.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)


# loss :  0.19143211841583252
# accuracy :  0.9241930246353149
# tf.Tensor([1 0 1 ... 5 1 1], shape=(116203,), dtype=int64)
# tf.Tensor([1 1 0 ... 5 1 1], shape=(116203,), dtype=int64)
# acc 스코어 :  0.9238315706134953

# dropout 사용 결과값
# loss :  0.3311381936073303
# accuracy :  0.8697279691696167
# tf.Tensor([1 0 1 ... 5 1 1], shape=(116203,), dtype=int64)
# tf.Tensor([1 1 0 ... 5 1 1], shape=(116203,), dtype=int64)
>>>>>>> 0032b7bc9af5be1bd054bebd127a94da7509d68b
# acc 스코어 :  0.8688846243212309