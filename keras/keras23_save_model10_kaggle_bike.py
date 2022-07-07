# 함수형 모델구성으로 변경

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
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

# print(train_set.shape) #(10886, 16)
# print(test_set.shape) #(6493, 15)
# print(x.shape) #(10886, 15)
# print(y.shape) #(10886,)


# [과제]
# # activation : sigmoid, relu, linear
# metrics 추가
# EarlyStopping 넣고
# 성능비교
# 느낀점 2줄 이상

x_train,x_test,y_train,y_test=train_test_split(x,y,
        train_size=0.99,shuffle=True, random_state=777)

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
# model=Sequential()
# model.add(Dense(32,input_dim=15))
# model.add(Dense(60,activation='ReLU'))
# model.add(Dense(100,activation='ReLU'))
# model.add(Dense(50,activation='ReLU'))
# model.add(Dense(30,activation='ReLU'))
# model.add(Dense(10,activation='ReLU'))
# model.add(Dense(1))

# input1 = Input(shape=(15,)) # 먼저 input layer를 명시해줌
# dense1 = Dense(32)(input1)
# dense2 = Dense(60, activation='relu')(dense1)
# dense3 = Dense(100, activation='relu')(dense2)
# dense4 = Dense(50, activation='relu')(dense3)
# dense5 = Dense(30, activation='relu')(dense4)
# dense6 = Dense(10, activation='relu')(dense5)
# output1 = Dense(1)(dense3)
# model = Model(inputs=input1, outputs=output1)


# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam',
#               metrics=['accuracy'])
# from tensorflow.python.keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, 
#                               restore_best_weights=True) 
# hist = model.fit(x_train, y_train, epochs=500, batch_size=100, 
#                 validation_split=0.2,
#                 callbacks=[earlyStopping],
#                 verbose=1)

# model.save("./_save/keras23_save_model10_kaggle_bike.h5")
model = load_model("./_save/keras23_save_model10_kaggle_bike.h5")

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

# y_summit=model.predict(test_set)
# print(y_summit)
# print(y_summit.shape)


# result=pd.read_csv(path+'sampleSubmission.csv',index_col=0)
# result['count']=y_summit
# result=abs(result)
# result.to_csv(path+'sampleSubmission.csv',index=True)

# loss: 2941.988525390625
# RMSE 54.24010049650571




# 1/ validation 적용
# loss: 5486.3203125
# RMSE 74.06970081950794

# 2/ EarlyStopping 및 activation 적용
# loss: [4765.76123046875, 0.0]
# RMSE 69.0344925448959

# 손실은 줄었으나 RMSE의 경우 감소한 것을 확인
# 조금 더 정교한 데이터 수정이 필요할 것으로 생각됨


#=============================================================================
# loss: [5437.6064453125, 0.0]
# RMSE 73.74012806742591
# R2 :  0.7540101816527908
#=============================================================================
# MinMaxScaler
# loss: [5129.3740234375, 0.0]
# RMSE 71.61964129290361
# R2 :  0.7679542386739596
#=============================================================================
# StandardScaler
# loss: [4965.01123046875, 0.0]
# RMSE 70.4628345382183
# R2 :  0.7753897474197846
#=============================================================================
# MaxAbsScaler
# loss: [4549.19384765625, 0.0]
# RMSE 67.44771321916453
# R2 :  0.7942007365136285
#=============================================================================
# RobustScaler
# loss: [5368.74365234375, 0.0]
# RMSE 73.27171552764139
# R2 :  0.7571254122056776

# 함수형 모델 =================================================================
# MaxAbsScaler
# loss: [4587.65185546875, 0.0]
# RMSE 67.7322071636783
# R2 :  0.792460955568764