# 예제 12개 만들고 최적의 weight 가중치 파일을 저장할 것

from pyexpat import model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sqlalchemy import false
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input

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

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # 수치로 변환해주는 걸 x_train에 집어넣자.
x_test = scaler.transform(x_test) 
# print(np.min(x_train)) # 0.0
# print(np.max(x_train)) # 1.0000000000000002
# print(np.min(x_test)) # -0.06141956477526944
# print(np.max(x_test)) # 1.1478180091225068

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# input1 = Input(shape=(13,)) # 먼저 input layer를 명시해줌
# dense1 = Dense(5)(input1)
# dense2 = Dense(9, activation='relu')(dense1)
# dense3 = Dense(10, activation='sigmoid')(dense2)
# dense4 = Dense(10, activation='relu')(dense3)
# output1 = Dense(1)(dense4)
# model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
# print(date) # 2022-07-07 17:21:37.577295 수치형 데이터
date = date.strftime("%m%d_%H%M")
print(date) # 0707_1723 자료형 데이터(문자형)

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

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

