# 캐글 문제 https://www.kaggle.com/datasets/pankrzysiu/weather-archive-jena
# 시계열 데이터, split 이용해 데이터를(컬럼) 뭉치로 잘라서 LSTM, Conv1D 이용해 RNN계열, CNN계열 하나씩 써서 evaluate 평가까지만 예측

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

#1. 데이터
path = './_data/kaggle_jena/' # 경로 = .현재폴더 /하단
data = pd.read_csv(path + 'jena_climate_2009_2016.csv',
                   index_col=0) # 0번째 컬럼은 인덱스로 지정
print(data)
print(data.shape) # (420551, 14)


# 데이터를 잘라주자.

def split_x(data, size):
    aaa = []
    for i in range(len(dataset) - size +1): # 10 - 5 + 1 = 6 / range 6 / range 횟수
        subset = dataset[i : (i + size)] # 1:6, 2:7, ... 이런식의 부분집합을 만들자.
        aaa.append(subset) # .append 선택된 요소의 마지막에 새로운 요소나 콘텐츠를 추가
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
# [[  1   2   3   4   5]
#  [  2   3   4   5   6]
#  ...
#   [ 95  96  97  98  99]
#  [ 96  97  98  99 100]]
print(bbb.shape) # (96, 5)



print(x_train.shape, x_test.shape) # (1095, 75) (365, 75)
print(y_train.shape, y_test.shape) # (1095,) (365,)
x_train = x_train.reshape(42551, 14, 1)
x_test = x_test.reshape(, 14, 1)
print(x_train.shape, x_test.shape) # (1095, 75, 1) (365, 75, 1)


#2. 모델구성

model = Sequential()
model.add(LSTM(64, input_shape=(30,1), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

model = Sequential()
model.add(Conv1D(64, 2, input_shape=(75, 1)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model = Sequential()
model.add(Conv1D(64, 2, padding='same', input_shape=(28, 28))) # (N, 28, 64)
model.add(MaxPooling1D()) # (N, 14, 64)
model.add(Conv1D(32, 2, padding='valid', activation='relu')) # (N, 13, 32)
model.add(Conv1D(32, 2, padding='valid', activation='relu')) # (N, 12, 32)
model.add(Flatten()) # (N, 12*32) = (N, 384)
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))





# # 데이터에서 5개의 기후관련 컬럼을 추출한 후,  데이터 30만개로 학습데이터 지정
# weather_data = np.array(data[['p (mbar)', 'T (degC)', 'VPmax (mbar)', 'sh (g/kg)', 'wv (m/s)']]) # 내부압력 파스칼, 섭씨온도, 포화증기압, 특정습도, 풍속
# train_data = weather_data[:300000]
# train_mean, train_std = np.mean(train_data), np.std(train_data)
# train_data = (train_data-train_mean)/train_std


train_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])

#print(train_set.columns)
#print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
#print(train_set.describe())



x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.75, shuffle=True, random_state=68)

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

print(x_train.shape, x_test.shape) # (1095, 75) (365, 75)
print(y_train.shape, y_test.shape) # (1095,) (365,)
x_train = x_train.reshape(1095, 75, 1)
x_test = x_test.reshape(365, 75, 1)
print(x_train.shape, x_test.shape) # (1095, 75, 1) (365, 75, 1)


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True) 
hist = model.fit(x_train, y_train, epochs=500, batch_size=100000, 
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

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('loss')
# plt.legend(['train_set', 'test_set'], loc='upper left')
# plt.show()

# print(y_summit)
# print(y_summit.shape) # (715, 1)


# .to_csv()를 사용하여
# submission.csv를 완성하시오

# sample_submission['SalePrice'] = y_summit
# sample_submission = sample_submission.fillna(sample_submission.mean())
# sample_submission.to_csv(path + 'test04.csv', index=True)


# LSTM
# loss : [163303.359375, 0.0]
# RMSE :  183232.58885887152
# R2 :  -5.530294328964465



# 1/ validation 적용
# loss : 15640.0517578125
# RMSE :  24100.263701740514

# 2/ EarlyStopping 및 activation 적용
# loss : [22665.880859375, 0.0]
# RMSE :  32121.92020250101

# 손실, RMSE 모두 증가한 것으로 확인
# 조금 더 정교한 데이터 수정이 필요할 것으로 생각됨



#=============================================================================
# loss : [22343.62890625, 0.0]
# RMSE :  32266.976806281906
# R2 :  0.7974912120556982
#=============================================================================
# MinMaxScaler
# loss : [18434.89453125, 0.0]
# RMSE :  27151.20760599753
# R2 :  0.8566143724201605
#=============================================================================
# StandardScaler
# loss : [16366.8154296875, 0.0]
# RMSE :  24108.475548483566
# R2 :  0.8869509754102445
#=============================================================================
# MaxAbsScaler
# # loss : [19213.369140625, 0.0]
# RMSE :  27788.8725431025
# R2 :  0.8498002623191562
#=============================================================================
# RobustScaler
# loss : [20941.740234375, 0.0]
# RMSE :  28724.662617243022
# R2 :  0.8395139849715223
'''