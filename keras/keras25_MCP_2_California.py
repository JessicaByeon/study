from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(20640, 8) (20640,)

print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
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
model.add(Dense(5, activation='linear', input_dim=8))
model.add(Dense(9, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(9, activation='linear'))
model.add(Dense(1, activation='linear'))

# input1 = Input(shape=(8,)) # 먼저 input layer를 명시해줌
# dense1 = Dense(5)(input1)
# dense2 = Dense(9, activation='linear')(dense1)
# dense3 = Dense(10, activation='relu')(dense2)
# dense4 = Dense(9, activation='linear')(dense2)
# output1 = Dense(1, activation='linear')(dense3)
# model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
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

# loss :  [0.43741923570632935, 0.0033909252379089594]
