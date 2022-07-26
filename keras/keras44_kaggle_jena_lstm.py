# 캐글 문제 https://www.kaggle.com/datasets/pankrzysiu/weather-archive-jena
# 시계열 데이터, split 이용해 데이터를(컬럼) 잘라서 LSTM, Conv1D 이용해 (RNN계열, CNN계열 하나씩) evaluate 평가까지만 예측

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
path = './_data/kaggle_jena/' # 경로 = .현재폴더 /하단
df_weather = pd.read_csv(path + 'jena_climate_2009_2016.csv',
                         index_col=0) # 0번째 컬럼은 인덱스로 지정
print(df_weather)
print(df_weather.shape) # (420551, 14)


# 데이터 정규화
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scale_cols = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
       'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
       'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
       'wd (deg)']
df_scaled = scaler.fit_transform(df_weather[scale_cols])

df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols

print(df_scaled)


# 학습을 시킬 데이터 셋 생성 / window_size를 정의하여 학습 데이터를 생성 
# window_size는 내가 얼마동안(기간)의 데이터에 기반하여 다음날 값을 예측할 것인가를 정하는 parameter 
# 즉 내가 과거 20일을 기반으로 내일 데이터를 예측한다라고 가정했을 때는 window_size=20
TEST_SIZE = 200 # 학습은 과거부터 200일 이전의 데이터 사용, TEST는 이후 200일의 데이터 사용

train = df_scaled[:-TEST_SIZE]
test = df_scaled[-TEST_SIZE:]

# dataset을 만들어 주는 함수

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

# 위의 함수는 정해진 window_size에 기반하여 20일 기간의 데이터 셋을 묶어 주는 역할
# 즉, 순차적으로 20일 동안의 데이터 셋을 묶고, 이에 맞는 label (예측 데이터)와 함께 return

# feature 와 label 정의
feature_cols = ['p (mbar)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
       'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
       'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
       'wd (deg)']
label_cols = ['T (degC)']

train_feature = train[feature_cols]
train_label = train[label_cols]

# train dataset
train_feature, train_label = make_dataset(train_feature, train_label, 20)

# train, validation set 생성
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

print(x_train.shape, x_valid.shape)
# (336264, 20, 13) (84067, 20, 13)

# test dataset (실제 예측 해볼 데이터)
test_feature = test[feature_cols]
test_label = test[label_cols]
test_feature, test_label = make_dataset(test_feature, test_label, 20)
print(test_feature.shape, test_label.shape)
# (180, 20, 13) (180, 1)

'''
# # 데이터에서 5개의 기후관련 컬럼을 추출한 후,  데이터 30만개로 학습데이터 지정
# weather_data = np.array(data[['p (mbar)', 'T (degC)', 'VPmax (mbar)', 'sh (g/kg)', 'wv (m/s)']]) # 내부압력 파스칼, 섭씨온도, 포화증기압, 특정습도, 풍속
# train_data = weather_data[:300000]
# train_mean, train_std = np.mean(train_data), np.std(train_data)
# train_data = (train_data-train_mean)/train_std

# shape 확인 후 reshape!
# print(x_train.shape, x_test.shape) # (1095, 75) (365, 75)
# print(y_train.shape, y_test.shape) # (1095,) (365,)
# x_train = x_train.reshape(42551, 14, 1)
# x_test = x_test.reshape(, 14, 1)
# print(x_train.shape, x_test.shape) # (1095, 75, 1) (365, 75, 1)
'''

#2. 모델구성
model = Sequential()
model.add(Conv1D(64, 3, padding='same',
                 input_shape=(20,13)))
model.add(MaxPooling1D())
model.add(LSTM(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
# model.summary


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True) 
hist = model.fit(x_train, y_train, epochs=100, batch_size=100000, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_valid, y_valid)
print('loss :', loss)


# Conv1D/LSTM
# 훈련량 1
# loss : [0.13782534003257751, 1.1895273928530514e-05]
# 훈련량 10
# loss : [0.07190541177988052, 1.1895273928530514e-05]
# 훈련량 100
# loss : [0.007612772285938263, 0.0]
# loss : [0.00676732836291194, 0.0]