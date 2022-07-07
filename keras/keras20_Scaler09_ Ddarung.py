<<<<<<< HEAD
# 아래 모델에 대해 3가지 비교

# 스케일링 하기 전
# MinMaxScaler
# StandardScaler

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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


# [과제]
# # activation : sigmoid, relu, linear
# metrics 추가
# EarlyStopping 넣고
# 성능비교
# 느낀점 2줄 이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.98, shuffle=True, random_state=68)

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
model.add(Dense(100, activation='linear', input_dim=9))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 

hist = model.fit(x_train, y_train, epochs=530, batch_size=100, 
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

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

# y_summit = model.predict(test_set)

# # print(y_summit)
# # print(y_summit.shape) # (715, 1)


# # .to_csv()를 사용하여
# # submission.csv를 완성하시오

# submission['count'] = y_summit
# submission = submission.fillna(submission.mean())
# submission.to_csv('test5.csv', index=True)

# loss : 16.401729583740234
# RMSE :  19.031272227452






# 1/ validation 적용
# loss : 26.81962013244629
# RMSE :  33.2320737256767

# 2/ EarlyStopping 및 activation 적용
# loss : [20.08526039123535, 0.0]
# RMSE :  26.914413185183808

# 손실은 줄었으나 RMSE의 경우 감소한 것을 확인
# 조금 더 정교한 데이터 수정이 필요할 것으로 생각됨




#=============================================================================
# loss : [27.84587860107422, 0.0]
# RMSE :  36.621039028702974
# R2 :  0.6128960671957424
#=============================================================================
# MinMaxScaler
# loss : [24.624011993408203, 0.0]
# RMSE :  31.756631348245385
# R2 :  0.7089047530358311
#=============================================================================
# StandardScaler
# loss : [25.760976791381836, 0.0]
# RMSE :  33.39240355312001
# R2 :  0.6781439924012276
#=============================================================================
# MaxAbsScaler
# loss : [24.063119888305664, 0.0]
# RMSE :  28.682953170013818
# R2 :  0.7625271376443159
#=============================================================================
# RobustScaler
# loss : [24.27006721496582, 0.0]
# RMSE :  32.281185067857685
# R2 :  0.6992087499501282
=======
# 아래 모델에 대해 3가지 비교

# 스케일링 하기 전
# MinMaxScaler
# StandardScaler

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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


# [과제]
# # activation : sigmoid, relu, linear
# metrics 추가
# EarlyStopping 넣고
# 성능비교
# 느낀점 2줄 이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.98, shuffle=True, random_state=68)

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
model.add(Dense(100, activation='linear', input_dim=9))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 

hist = model.fit(x_train, y_train, epochs=530, batch_size=100, 
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

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

# y_summit = model.predict(test_set)

# # print(y_summit)
# # print(y_summit.shape) # (715, 1)


# # .to_csv()를 사용하여
# # submission.csv를 완성하시오

# submission['count'] = y_summit
# submission = submission.fillna(submission.mean())
# submission.to_csv('test5.csv', index=True)

# loss : 16.401729583740234
# RMSE :  19.031272227452






# 1/ validation 적용
# loss : 26.81962013244629
# RMSE :  33.2320737256767

# 2/ EarlyStopping 및 activation 적용
# loss : [20.08526039123535, 0.0]
# RMSE :  26.914413185183808

# 손실은 줄었으나 RMSE의 경우 감소한 것을 확인
# 조금 더 정교한 데이터 수정이 필요할 것으로 생각됨




#=============================================================================
# loss : [27.84587860107422, 0.0]
# RMSE :  36.621039028702974
# R2 :  0.6128960671957424
#=============================================================================
# MinMaxScaler
# loss : [24.624011993408203, 0.0]
# RMSE :  31.756631348245385
# R2 :  0.7089047530358311
#=============================================================================
# StandardScaler
# loss : [25.760976791381836, 0.0]
# RMSE :  33.39240355312001
# R2 :  0.6781439924012276
#=============================================================================
# MaxAbsScaler
# loss : [24.063119888305664, 0.0]
# RMSE :  28.682953170013818
# R2 :  0.7625271376443159
#=============================================================================
# RobustScaler
# loss : [24.27006721496582, 0.0]
# RMSE :  32.281185067857685
# R2 :  0.6992087499501282
>>>>>>> 425a270641f2121145cc0df2169ab516031dcf7e
