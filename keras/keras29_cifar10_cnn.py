# 컬러, 채널값 3
# 분류값 10개

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout # 이미지 작업은 2D
from keras.datasets import mnist, cifar10
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


#1. 데이터

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = StandardScaler()
# scaler.fit(x_train)
# scaler.transform(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)
print(x_train.shape) # (50000, 32, 32, 3)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
#  array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

# reshape 할 때 모든 개체를 곱한 값은 동일해야한다.
# 모양은 바꿀 수 있다. 다만 데이터 순서만 바뀌지 않으면 됨

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), # 64 다음 레이어로 전달해주는 아웃풋 노드의 갯수, kernel size 이미지를 자르는 규격
                 padding='same', # 원래 shape를 그대로 유지하여 다음 레이어로 보내주고 싶을 때 주로 사용!
                 input_shape=(32, 32, 3))) # (batch_size, rows, columns, channels)   # 출력 : (N, 32, 32, 64)
model.add(MaxPooling2D()) # (N, 16, 16, 64)
model.add(Conv2D(32, (2,2), 
                 padding='valid', # 디폴트
                 activation='relu')) # filter = 32, kernel size = (2,2) # 출력 : (N, 15, 15, 32)
model.add(Conv2D(32, (2,2), 
                 padding='valid', # 디폴트
                 activation='relu')) # filter = 32, kernel size = (2,2) # 출력 : (N, 14, 14, 32)
model.add(Flatten()) # (N, 6272) 14*14*32
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
# print(date) # 2022-07-07 17:21:37.577295 수치형 데이터
date = date.strftime("%m%d_%H%M")
print(date) # 0707_1723 자료형 데이터(문자형)

# 파일명을 계속적으로 수정하지 않고 고정시켜주기 위해
filepath = './_ModelCheckPoint/k28/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # d4 네자리까지, .4f 소수넷째자리까지

earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
                      save_best_only=True,
                      filepath= "".join([filepath, 'k28_', date, '_', filename] # .join안에 있는 모든 문자열을 합치겠다.
                      ))

# start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=200, 
                validation_split=0.2,
                callbacks=[earlyStopping, mcp], # 최저값을 체크해 반환해줌
                verbose=1)
# end_time = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, accuracy_score
y_predict = np.argmax(y_predict, axis= 1)
y_predict = to_categorical(y_predict)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)


# loss :  [1.3300107717514038, 0.5730000138282776]
# acc스코어 :  0.573

