<<<<<<< HEAD
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout

import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,) 8*8 이미지가 1797개 있다는..
print(np.unique(y, return_counts=True)) 
# [0 1 2 3 4 5 6 7 8 9] dim 10 softmax (1797,10)으로 원핫인코딩 평가지표 accuracy
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
# print(datasets.DESCR)
# print(datasets.feature_names)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) #(1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # 수치로 변환해주는 걸 x_train에 집어넣자.
x_test = scaler.transform(x_test) 
# print(np.min(x_train))
# print(np.max(x_train))
# print(np.min(x_test))
# print(np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='linear', input_dim=64))
model.add(Dropout(0.3)) # 30% 만큼 제외
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(10, activation='linear'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(10, activation='softmax'))

# input1 = Input(shape=(64,)) # 먼저 input layer를 명시해줌
# dense1 = Dense(5, activation='linear')(input1)
# dense2 = Dense(10, activation='relu')(dense1)
# dense3 = Dense(10, activation='relu')(dense2)
# dense3 = Dense(10, activation='linear')(dense2)
# output1 = Dense(10, activation='softmax')(dense3)
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

earlyStopping =EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, 
                             restore_best_weights=True) 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
                      save_best_only=True,
                      filepath= "".join([filepath, 'k24_', date, '_', filename] # .join안에 있는 모든 문자열을 합치겠다.
                      ))

hist = model.fit(x_train, y_train, epochs=100, batch_size=100, 
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

# print("============= y_test[:5] ==============")
# print(y_test[:5])
# print("============= y_pred ==============")
# y_predict = model.predict(x_test[:5])
# print(y_predict)
print("============= y_pred ==============")

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)


# loss :  0.39301151037216187
# accuracy :  0.8888888955116272

# dropout 사용 결과값
# loss :  0.8342002034187317
=======
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout

import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,) 8*8 이미지가 1797개 있다는..
print(np.unique(y, return_counts=True)) 
# [0 1 2 3 4 5 6 7 8 9] dim 10 softmax (1797,10)으로 원핫인코딩 평가지표 accuracy
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
# print(datasets.DESCR)
# print(datasets.feature_names)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) #(1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # 수치로 변환해주는 걸 x_train에 집어넣자.
x_test = scaler.transform(x_test) 
# print(np.min(x_train))
# print(np.max(x_train))
# print(np.min(x_test))
# print(np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='linear', input_dim=64))
model.add(Dropout(0.3)) # 30% 만큼 제외
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(10, activation='linear'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(10, activation='softmax'))

# input1 = Input(shape=(64,)) # 먼저 input layer를 명시해줌
# dense1 = Dense(5, activation='linear')(input1)
# dense2 = Dense(10, activation='relu')(dense1)
# dense3 = Dense(10, activation='relu')(dense2)
# dense3 = Dense(10, activation='linear')(dense2)
# output1 = Dense(10, activation='softmax')(dense3)
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

earlyStopping =EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, 
                             restore_best_weights=True) 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
                      save_best_only=True,
                      filepath= "".join([filepath, 'k24_', date, '_', filename] # .join안에 있는 모든 문자열을 합치겠다.
                      ))

hist = model.fit(x_train, y_train, epochs=100, batch_size=100, 
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

# print("============= y_test[:5] ==============")
# print(y_test[:5])
# print("============= y_pred ==============")
# y_predict = model.predict(x_test[:5])
# print(y_predict)
print("============= y_pred ==============")

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)


# loss :  0.39301151037216187
# accuracy :  0.8888888955116272

# dropout 사용 결과값
# loss :  0.8342002034187317
>>>>>>> 0032b7bc9af5be1bd054bebd127a94da7509d68b
# accuracy :  0.7194444537162781