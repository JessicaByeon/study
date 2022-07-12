from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR) # (569, 30)
# print(datasets.feature_names)

x = datasets['data'] # datasets.data 로도 쓸 수 있음
y = datasets['target']
print(x.shape, y.shape) # (569, 30) (569,)

# print(x)
# print(y) # 0, 1이 569개

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.9, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # 수치로 변환해주는 걸 x_train에 집어넣자.
x_test = scaler.transform(x_test) 
# print(np.min(x_train))
# print(np.max(x_train))
# print(np.min(x_test))
# print(np.max(x_test))


#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='linear', input_dim=30))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

# input1 = Input(shape=(30,)) # 먼저 input layer를 명시해줌
# dense1 = Dense(5, activation='linear')(input1)
# dense2 = Dense(10, activation='sigmoid')(dense1)
# dense3 = Dense(10, activation='relu')(dense2)
# dense4 = Dense(10, activation='linear')(dense3)
# output1 = Dense(1, activation='sigmoid')(dense4)
# model = Model(inputs=input1, outputs=output1)


# #3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
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

hist = model.fit(x_train, y_train, epochs=100, batch_size=20, 
                validation_split=0.2,
                callbacks=[earlyStopping, mcp],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# print('------------------------------')
# print(hist) # <tensorflow.python.keras.callbacks.History object at 0x00000219A7310F40>
# print('------------------------------')
# print(hist.history) 
# print('------------------------------')
# print(hist.history['loss']) #키밸류 상의 loss는 이름이기 때문에 ''를 넣어줌
# print('------------------------------')
# print(hist.history['val_loss']) #키밸류 상의 val_loss는 이름이기 때문에 ''를 넣어줌

# print("걸린시간 : ", end_time)


# 그래프 그리기 전에 r2/acc
y_predict = model.predict(x_test)
y_predict[(y_predict<0.5)] = 0  
y_predict[(y_predict>=0.5)] = 1 
# print(y_predict)
# print(y_predict)

#### [과제 1.] accuracy score 완성

from sklearn.metrics import r2_score, accuracy_score
# # r2 = r2_score(y_test, y_predict)
# # print('r2 스코어 : ', r2)
# # r2 스코어 :  0.5852116219896948

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)
print(y_predict)

# loss :  [0.04178836569190025, 0.9824561476707458]
# acc 스코어 :  0.9824561403508771