from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
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

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train) # 위 2줄을 이 한줄로 표현가능 fit.transform
x_test = scaler.transform(x_test) # x_train은 fit, transform 모두 실행, x_test는 transform만! fix X!

print(x_train.shape) # (14447, 8)
print(x_test.shape) # (6193, 8)

x_train = x_train.reshape(14447, 2, 2, 2)
x_test = x_test.reshape(6193, 2, 2, 2)
print(np.unique(y_train, return_counts=True))


#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='linear', input_dim=8))
model.add(Dropout(0.3)) # 30% 만큼 제외
model.add(Dense(9, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(9, activation='linear'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(1, activation='linear'))

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), # 64 다음 레이어로 전달해주는 아웃풋 노드의 갯수, kernel size 이미지를 자르는 규격
                 padding='same', # 원래 shape를 그대로 유지하여 다음 레이어로 보내주고 싶을 때 주로 사용!
                 input_shape=(2, 2, 2))) # (batch_size, rows, columns, channels) / 출력 (N, 2, 2, 64)
model.add(Conv2D(32, (2,2), 
                 padding='same',
                 activation='relu')) # filter = 32, kernel size = (2,2) / 출력 (N, 2, 2, 32)
model.add(Flatten()) # (N, 128)
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3)) # 30% 만큼 제외
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# date = datetime.datetime.now()
# # print(date)
# date = date.strftime("%m%d_%H%M")
# # print(date) 

# # 파일명을 계속적으로 수정하지 않고 고정시켜주기 위해
# filepath = './_ModelCheckPoint/k24/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # d4 네자리까지, .4f 소수넷째자리까지

earlyStopping =EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, 
                             restore_best_weights=True) 

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
#                       save_best_only=True,
#                       filepath= "".join([filepath, 'k24_', date, '_', filename] # .join안에 있는 모든 문자열을 합치겠다.
#                       ))

hist = model.fit(x_train, y_train, epochs=100, batch_size=200, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)
# loss :  [0.43741923570632935, 0.0033909252379089594]


# dropout 사용 결과값
# loss :  [0.7747023105621338, 0.0033909252379089594]

# cnn ==============================================================================
# MinMaxScale
# loss :  [0.4570671617984772, 0.0033909252379089594]    
# r2 스코어 :  0.6669019412529609
