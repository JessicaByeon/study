<<<<<<< HEAD
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/ddarung/' # 경로 = .현재폴더 /하단
train_set = pd.read_csv(path + 'train.csv', # train.csv 의 데이터가 train set에 들어가게 됨
                        index_col=0) # 0번째 컬럼은 인덱스로 지정하는 명령
print(train_set)
print(train_set.shape) # (1459, 10) 원래 열이 11개지만, id를 인덱스로 제외하여 10개

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)
submission = pd.read_csv(path + 'submission.csv',
                       index_col=0)
print(test_set)
print(test_set.shape) # (715, 9) # 예측 과정에서 쓰일 예정

print(train_set.columns)
print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
print(train_set.describe())

# 결측치가 있어 데이터를 계산 시 nan/null 값이 되므로 결측치를 삭제 해 주면 됨
# 결측치 삭제를 위해 같은 행을 삭제해버릴경우, 해당 행의 기존에 있던 데이터 들이 사라지므로 위험 

#### 결측치 처리 1. 제거 ####
print(train_set.isnull().sum()) # 각 컬럼당 null의 갯수 확인가능
train_set = train_set.fillna(train_set.mean()) # nan 값을 행별로 모두 삭제(dropna)
print(train_set.isnull().sum())
print(train_set.shape) # (1328, 10) 데이터가 얼마나 삭제된 것인지 확인가능 1459-1328=131개 삭제


test_set = test_set.fillna(test_set.mean())


x = train_set.drop(['count'], axis=1) # axis는 'count'가 컬럼이라는 것을 명시하기 위해
print(x)
print(x.columns)
print(x.shape) # (1459, 9)

y = train_set['count']
print(y)
print(y.shape) # (1459,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.98, shuffle=True, random_state=68)

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

print(x_train.shape) # (1429, 9)
print(x_test.shape) # (30, 9)

x_train = x_train.reshape(1429, 3, 3, 1)
x_test = x_test.reshape(30, 3, 3, 1)
print(np.unique(y_train, return_counts=True))


#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), # 64 다음 레이어로 전달해주는 아웃풋 노드의 갯수, kernel size 이미지를 자르는 규격
                 padding='same', # 원래 shape를 그대로 유지하여 다음 레이어로 보내주고 싶을 때 주로 사용!
                 input_shape=(3, 3, 1))) # (batch_size, rows, columns, channels) / 출력 (N, 3, 3, 64)
model.add(Conv2D(32, (2,2), 
                 padding='same',
                 activation='relu')) # filter = 32, kernel size = (2,2) / 출력 (N, 3, 3, 32)
model.add(Flatten()) # (N, 288)
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3)) # 30% 만큼 제외
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# date = datetime.datetime.now()
# # print(date)
# date = date.strftime("%m%d_%H%M")
# # print(date) 

#  # 파일명을 계속적으로 수정하지 않고 고정시켜주기 위해
# filepath = './_ModelCheckPoint/k24/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # d4 네자리까지, .4f 소수넷째자리까지

earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
#                       save_best_only=True,
#                       filepath= "".join([filepath, 'k24_', date, '_', filename] # .join안에 있는 모든 문자열을 합치겠다.
#                       ))

hist = model.fit(x_train, y_train, epochs=530, batch_size=100, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : #(원y값, 예측y값)
    return np.sqrt(mean_squared_error(y_test, y_predict)) # MSE에 루트를 씌워 돌려주겠다.

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# loss : [24.83771324157715, 0.0]
# RMSE :  31.007347425685435
# R2 :  0.7224792311036958

# dropout 사용 결과값
# loss : [25.49931526184082, 0.0]
# RMSE :  33.80476547609509
# R2 :  0.6701457273411999

# cnn ==============================================================================
# MaxAbsScaler
# loss : [27.15427017211914, 0.0]
# RMSE :  34.757137567351286
# R2 :  0.6512981391961696
=======
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/ddarung/' # 경로 = .현재폴더 /하단
train_set = pd.read_csv(path + 'train.csv', # train.csv 의 데이터가 train set에 들어가게 됨
                        index_col=0) # 0번째 컬럼은 인덱스로 지정하는 명령
print(train_set)
print(train_set.shape) # (1459, 10) 원래 열이 11개지만, id를 인덱스로 제외하여 10개

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)
submission = pd.read_csv(path + 'submission.csv',
                       index_col=0)
print(test_set)
print(test_set.shape) # (715, 9) # 예측 과정에서 쓰일 예정

print(train_set.columns)
print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
print(train_set.describe())

# 결측치가 있어 데이터를 계산 시 nan/null 값이 되므로 결측치를 삭제 해 주면 됨
# 결측치 삭제를 위해 같은 행을 삭제해버릴경우, 해당 행의 기존에 있던 데이터 들이 사라지므로 위험 

#### 결측치 처리 1. 제거 ####
print(train_set.isnull().sum()) # 각 컬럼당 null의 갯수 확인가능
train_set = train_set.fillna(train_set.mean()) # nan 값을 행별로 모두 삭제(dropna)
print(train_set.isnull().sum())
print(train_set.shape) # (1328, 10) 데이터가 얼마나 삭제된 것인지 확인가능 1459-1328=131개 삭제


test_set = test_set.fillna(test_set.mean())


x = train_set.drop(['count'], axis=1) # axis는 'count'가 컬럼이라는 것을 명시하기 위해
print(x)
print(x.columns)
print(x.shape) # (1459, 9)

y = train_set['count']
print(y)
print(y.shape) # (1459,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.98, shuffle=True, random_state=68)

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

print(x_train.shape) # (1429, 9)
print(x_test.shape) # (30, 9)

x_train = x_train.reshape(1429, 3, 3, 1)
x_test = x_test.reshape(30, 3, 3, 1)
print(np.unique(y_train, return_counts=True))


#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), # 64 다음 레이어로 전달해주는 아웃풋 노드의 갯수, kernel size 이미지를 자르는 규격
                 padding='same', # 원래 shape를 그대로 유지하여 다음 레이어로 보내주고 싶을 때 주로 사용!
                 input_shape=(3, 3, 1))) # (batch_size, rows, columns, channels) / 출력 (N, 3, 3, 64)
model.add(Conv2D(32, (2,2), 
                 padding='same',
                 activation='relu')) # filter = 32, kernel size = (2,2) / 출력 (N, 3, 3, 32)
model.add(Flatten()) # (N, 288)
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3)) # 30% 만큼 제외
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# date = datetime.datetime.now()
# # print(date)
# date = date.strftime("%m%d_%H%M")
# # print(date) 

#  # 파일명을 계속적으로 수정하지 않고 고정시켜주기 위해
# filepath = './_ModelCheckPoint/k24/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # d4 네자리까지, .4f 소수넷째자리까지

earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
#                       save_best_only=True,
#                       filepath= "".join([filepath, 'k24_', date, '_', filename] # .join안에 있는 모든 문자열을 합치겠다.
#                       ))

hist = model.fit(x_train, y_train, epochs=530, batch_size=100, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : #(원y값, 예측y값)
    return np.sqrt(mean_squared_error(y_test, y_predict)) # MSE에 루트를 씌워 돌려주겠다.

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# loss : [24.83771324157715, 0.0]
# RMSE :  31.007347425685435
# R2 :  0.7224792311036958

# dropout 사용 결과값
# loss : [25.49931526184082, 0.0]
# RMSE :  33.80476547609509
# R2 :  0.6701457273411999

# cnn ==============================================================================
# MaxAbsScaler
# loss : [27.15427017211914, 0.0]
# RMSE :  34.757137567351286
# R2 :  0.6512981391961696
>>>>>>> 0032b7bc9af5be1bd054bebd127a94da7509d68b
