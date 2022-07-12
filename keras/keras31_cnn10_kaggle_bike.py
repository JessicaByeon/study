from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from csv import reader
from pandas import DataFrame

#.1 데이터
path='./_data/kaggle_bike/'
train_set=pd.read_csv(path+'train.csv')
test_set=pd.read_csv(path+'test.csv') #예측할때 사용할거에요!!


#데이트 타임 연/월/일/시 로 컬럼 나누기
train_set['datetime']=pd.to_datetime(train_set['datetime']) #date time 열을 date time 속성으로 변경
#세부 날짜별 정보를 보기 위해 날짜 데이터를 년도, 월, 일, 시간으로 나눠준다.(분,초는 모든값이 0 이므로 추가하지않는다.)
train_set['year']=train_set['datetime'].dt.year
train_set['month']=train_set['datetime'].dt.month
train_set['day']=train_set['datetime'].dt.day
train_set['hour']=train_set['datetime'].dt.hour

#날짜와 시간에 관련된 피쳐에는 datetime, holiday, workingday,year,month,day,hour 이 있다.
#숫자형으로 나오는 holiday,workingday,month,hour만 쓰고 나머지 제거한다.

train_set.drop(['datetime','day','year'],inplace=True,axis=1) #datetime, day, year 제거하기

#month, hour은 범주형으로 변경해주기
train_set['month']=train_set['month'].astype('category')
train_set['hour']=train_set['hour'].astype('category')

#season과 weather은 범주형 피쳐이다. 두 피쳐 모두 숫자로 표현되어 있으니 문자로 변환해준다.
train_set=pd.get_dummies(train_set,columns=['season','weather'])

#casual과 registered는 test데이터에 존재하지 않기에 삭제한다.
train_set.drop(['casual', 'registered'], inplace=True, axis=1)
#temp와 atemp는 상관관계가 아주 높고 두 피쳐의 의미가 비슷하기 때문에 temp만 사용한다.
train_set.drop('atemp',inplace=True,axis=1) #atemp 지우기

#위처럼 test_set도 적용하기
test_set['datetime']=pd.to_datetime(test_set['datetime'])

test_set['month']=test_set['datetime'].dt.month
test_set['hour']=test_set['datetime'].dt.hour

test_set['month']=test_set['month'].astype('category')
test_set['hour']=test_set['hour'].astype('category')

test_set=pd.get_dummies(test_set,columns=['season','weather'])

drop_feature = ['datetime', 'atemp']
test_set.drop(drop_feature, inplace=True, axis=1)

x = train_set.drop(['count'], axis=1)
y=train_set['count']

print(train_set.shape) #(10886, 16)
print(test_set.shape) #(6493, 15)
print(x.shape) #(10886, 15)
print(y.shape) #(10886,)

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99, shuffle=True, random_state=777)

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

print(x_train.shape) # (10777, 15)
print(x_test.shape) # (109, 15)

x_train = x_train.reshape(10777, 5, 3, 1)
x_test = x_test.reshape(109, 5, 3, 1) 

print(np.unique(y_train, return_counts=True))


#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), # 64 다음 레이어로 전달해주는 아웃풋 노드의 갯수, kernel size 이미지를 자르는 규격
                 padding='same', # 원래 shape를 그대로 유지하여 다음 레이어로 보내주고 싶을 때 주로 사용!
                 input_shape=(5, 3, 1))) # (batch_size, rows, columns, channels) / 출력 (N, 5, 3, 64)
model.add(Conv2D(32, (2,2), 
                 padding='same',
                 activation='relu')) # filter = 32, kernel size = (2,2) / 출력 (N, 5, 3, 32)
model.add(Flatten()) # (N, 480)
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

#  # 파일명을 계속적으로 수정하지 않고 고정시켜주기 위해
# filepath = './_ModelCheckPoint/k24/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # d4 네자리까지, .4f 소수넷째자리까지

earlyStopping =EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, 
                             restore_best_weights=True) 

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
#                       save_best_only=True,
#                       filepath= "".join([filepath, 'k24_', date, '_', filename] # .join안에 있는 모든 문자열을 합치겠다.
#                       ))

hist = model.fit(x_train, y_train, epochs=100, batch_size=100, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음


#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss:',loss)

y_predict=model.predict(x_test)



def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

rmse=RMSE(y_test,y_predict)
print("RMSE",rmse)

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


# loss: [4543.44775390625, 0.0]
# RMSE 67.40510079044358
# R2 :  0.7944606959926506

# dropout 사용 결과값
# loss: [4740.1865234375, 0.0]
# RMSE 68.84901377448787
# R2 :  0.7855604911084013

# cnn ==============================================================================
# MaxAbsScaler()
# loss: [8688.7158203125, 0.0]
# RMSE 93.21328344083268
# R2 :  0.6069344614994112