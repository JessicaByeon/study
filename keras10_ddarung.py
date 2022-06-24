# 데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
print(test_set)
print(test_set.shape) # (715, 9) # 예측 과정에서 쓰일 예정

print(train_set.columns)
print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
print(train_set.describe())

# 결측치가 있어 데이터를 계산 시 nan/null 값이 되므로 결측치를 삭제 해 주면 됨
# 결측치 삭제를 위해 같은 행을 삭제해버릴경우, 해당 행의 기존에 있던 데이터 들이 사라지므로 위험 

#### 결측치 처리 1. 제거 ####
print(train_set.isnull().sum()) # 각 컬럼당 null의 갯수 확인가능
train_set = train_set.dropna() # nan 값을 행별로 모두 삭제
print(train_set.isnull().sum())
print(train_set.shape) # (1328, 10) 데이터가 얼마나 삭제된 것인지 확인가능 1459-1328=131개 삭제

x = train_set.drop(['count'], axis=1) # axis는 'count'가 컬럼이라는 것을 명시하기 위해
print(x)
print(x.columns)
print(x.shape) # (1459, 9)

y = train_set['count']
print(y)
print(y.shape) # (1459,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7, shuffle=True, random_state=90)

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=9))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=250, batch_size=42)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : #(원y값, 예측y값)
    return np.sqrt(mean_squared_error(y_test, y_predict)) # MSE에 루트를 씌워 돌려주겠다.

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# loss : 2456.93212890625
# RMSE :  49.56745103837174
