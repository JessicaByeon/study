# 데이콘 쇼핑몰 지점별 매출액 예측 문제풀이 (각 지점별 주간 매출액 예측)
# https://dacon.io/competitions/official/235942/overview/description

import numpy as np
import pandas as pd
from sympy import linear_eq_to_matrix
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


#1. 데이터
path = './_data/dacon_shopping/' # 경로 = .현재폴더 /하단
train_set = pd.read_csv(path + 'train.csv', # train.csv 의 데이터가 train set에 들어가게 됨
                        index_col=0) # 0번째 컬럼은 인덱스로 지정하는 명령
print(train_set)
print(train_set.shape) # (6255, 12) 원래 열이 13개지만, id를 인덱스로 제외하여 12개

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)
submission = pd.read_csv(path + 'submission.csv',
                       index_col=0)
print(test_set)
print(test_set.shape) # (180, 11) # 예측 과정에서 쓰일 예정

print(train_set.columns)
# Index(['Store', 'Date', 'Temperature', 'Fuel_Price', 'Promotion1',
#        'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment',
#        'IsHoliday', 'Weekly_Sales'],
#       dtype='object')
print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
print(train_set.describe())
# Data columns (total 12 columns):
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   Store         6255 non-null   int64
#  1   Date          6255 non-null   object
#  2   Temperature   6255 non-null   float64
#  3   Fuel_Price    6255 non-null   float64
#  4   Promotion1    2102 non-null   float64
#  5   Promotion2    1592 non-null   float64
#  6   Promotion3    1885 non-null   float64
#  7   Promotion4    1819 non-null   float64
#  8   Promotion5    2115 non-null   float64
#  9   Unemployment  6255 non-null   float64
#  10  IsHoliday     6255 non-null   bool
#  11  Weekly_Sales  6255 non-null   float64
# dtypes: bool(1), float64(9), int64(1), object(1)       
# memory usage: 592.5+ KB
# None
#              Store  ...  Weekly_Sales
# count  6255.000000  ...  6.255000e+03
# mean     23.000000  ...  1.047619e+06
# std      12.988211  ...  5.654362e+05
# min       1.000000  ...  2.099862e+05
# 25%      12.000000  ...  5.538695e+05
# 50%      23.000000  ...  9.604761e+05
# 75%      34.000000  ...  1.421209e+06
# max      45.000000  ...  3.818686e+06
# [8 rows x 10 columns]

# 프로모션 1~5 컬럼에 결측치가 존재함을 확인
# date(날짜, 일/월/년), isholiday(T/F) 부분은 숫자가 아니기 때문에 분석 전에 데이터 전처리가 필요

# 결측치가 있어 데이터를 계산 시 nan/null 값이 되므로 결측치를 삭제 해 주면 됨
# 결측치 삭제를 위해 같은 행을 삭제해버릴경우, 해당 행의 기존에 있던 데이터 들이 사라지므로 위험 

# ===== 결측치 처리 =====
print(train_set.isnull().sum()) # 각 컬럼당 null의 갯수 확인가능
# Promotion1      4153
# Promotion2      4663
# Promotion3      4370
# Promotion4      4436
# Promotion5      4140

# import matplotlib.pyplot as plt
# # 예측값인 Weekly_Sales를 확인
# plt.hist(train_set.Weekly_Sales, bins=50)
# plt.show()
# # 매출이 0.2~2.2 사이(초반구간)에 밀집된 것을 확인

# train_set = train_set.fillna(train_set.mean()) # nan 값을 평균값으로 채움
# # print(train_set.isnull().sum())
# # print(train_set.shape) # (6255, 11) 데이터가 모두 채워진 것을 확인 가능
# test_set = test_set.fillna(test_set.mean())

train_set = train_set.fillna(0) # nan 값을 0으로 채움
# print(train_set.isnull().sum())
# print(train_set.shape) # (6255, 11) 데이터가 모두 채워진 것을 확인 가능
test_set = test_set.fillna(0)

train_set['Date'] = pd.to_datetime(train_set['Date'])
train_set['year'] = train_set['Date'].dt.year
train_set['month'] = train_set['Date'].dt.month
train_set['day'] = train_set['Date'].dt.day
#print(train_set) #확인.

test_set['Date'] = pd.to_datetime(test_set['Date'])
test_set['year'] = test_set['Date'].dt.year
test_set['month'] = test_set['Date'].dt.month
test_set['hour'] = test_set['Date'].dt.day
#print(test_set) #확인.


train_set = train_set.drop(columns=['Date'])
test_set = test_set.drop(columns=['Date'])

'''
# 날짜에 해당하는 문자를 숫자(월)로 추출
# Date 칼럼에서 "월"에 해당하는 정보만 추출 -> 숫자 형태로 반환하는 함수 작성
def get_month(date):
    month = date[3:5]
    month = int(month)
    return month

# 이 함수를 Date 칼럼에 적용한 Month 칼럼 생성
train_set['Month'] = train_set['Date'].apply(get_month)
print(train_set) # 월만 추출된 상태
test_set['Month'] = test_set['Date'].apply(get_month) # test_set 에도 적용
'''

# boolean 0 or 1로 변환
train_set["IsHoliday"] = train_set["IsHoliday"].astype(int)
test_set["IsHoliday"] = test_set["IsHoliday"].astype(int)

# # 분석할 의미가 없는 칼럼 제거
# train = train.drop(columns=['id'])
# test = test.drop(columns=['id'])

# 전처리 하기 전 칼럼 제거
train_set = train_set.drop(columns=['IsHoliday'])
test_set = test_set.drop(columns=['IsHoliday'])

# x = train_set.drop(['Date', 'IsHoliday'], axis=1) # axis는 'count'가 컬럼이라는 것을 명시하기 위해
print(train_set)
print(train_set.columns)
print(train_set.shape) # (6255, 11)
# Index(['Store', 'Temperature', 'Fuel_Price', 'Promotion1', 
# 'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment', 'Weekly_Sales', 'Month'], dtype='object')

# 학습에 사용할 정보와 예측하고자 하는 정보를 분리
x = train_set.drop(columns=['Weekly_Sales'])
print(x.shape) # (6255, 10)

y = train_set[['Weekly_Sales']]

# y = train_set['count']
print(y)
print(y.shape) # (6255, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.9, shuffle=True, random_state=100)

scaler = MinMaxScaler()
# scaler = StandardScaler()
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
model.add(Dense(100, input_dim=12))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# import time
#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
                      save_best_only=True,
                      filepath='./_ModelCheckPoint/keras32_ModelCheckPoint.hdf5' # 가장 낮은 지점이 이 경로에 저장, 낮은 값이 나올 때마다 계속적으로 갱신하여 저장
                      )

# start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=500, 
                validation_split=0.2,
                callbacks=[earlyStopping, mcp], # 최저값을 체크해 반환해줌
                verbose=1)
# end_time = time.time()


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : #(원y값, 예측y값)
    return np.sqrt(mean_squared_error(y_test, y_predict)) # MSE에 루트를 씌워 돌려주겠다.

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

y_summit = model.predict(test_set)

# print(y_summit)
# print(y_summit.shape)

# .to_csv()를 사용하여
# submission.csv를 완성하시오

submission['Weekly_Sales'] = y_summit
# submission = submission.fillna(submission.mean())
submission.to_csv(path + 'test01.csv', index=True)


# MinMaxScaler random_state 68--- 평균적으로 가장 결과값이 좋은 상태
# loss : 394664.25
# RMSE :  501303.8461835106
# r2 스코어 :  0.1851752432148115

# StandardScaler random_state 68
# loss : [401021.4375, 250133397504.0]
# RMSE :  500133.34311958484
# r2 스코어 :  0.17982951092451094

# MaxAbsScaler random_state 68
# loss : [405057.8125, 247482155008.0]
# RMSE :  497475.76673653687
# r2 스코어 :  0.18852269112536457

# RobustScaler random_state 68
# loss : [423380.8125, 292641701888.0]
# RMSE :  540963.6680906413
# r2 스코어 :  0.04044755442941761

# ===========================================
# 모델링 수정 후
# MinMaxScaler random_state 100
# RMSE :  498139.23646465497
# r2 스코어 :  0.2280599623524645
