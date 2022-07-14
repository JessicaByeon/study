from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM
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

# [과제]
# # activation : sigmoid, relu, linear
# metrics 추가
# EarlyStopping 넣고
# 성능비교
# 느낀점 2줄 이상

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

print(x_train.shape, x_test.shape) # (14447, 8) (6193, 8)
print(y_train.shape, y_test.shape) # (14447,) (6193,)
x_train = x_train.reshape(14447, 8, 1)
x_test = x_test.reshape(6193, 8, 1)
print(x_train.shape, x_test.shape) # (14447, 8, 1) (6193, 8, 1)


#2. 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(8,1), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 
# earlyStopping 보통 변수는 앞글자 소문자
# 모니터 val_loss 대신 loss도 가능

# start_time = time.time() # 현재 시간 출력
hist = model.fit(x_train, y_train, epochs=10, batch_size=20, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1)
# end_time = time.time() - start_time # 걸린 시간

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# print("걸린시간 : ", end_time)

# 그래프 그리기 전에 r2
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# LSTM
# loss :  [0.6212025284767151, 0.0033909252379089594]        
# r2 스코어 :  0.5472844885084793






# [patience=10 일때]
# loss :  0.7213335037231445
# r2 스코어 :  0.47431188835527194
# [patience=100 일때]
# loss :  0.6468518972396851
# r2 스코어 :  0.5285920019123878


#=============================================================================
# loss :  [1.6712443828582764, 0.0032294525299221277]
# r2 스코어 :  -0.2179573696634438
#=============================================================================
# MinMaxScaler
# loss : [0.47200360894203186, 0.0033909252379089594]
# r2 스코어 :  0.6560166670365419
#=============================================================================
# StandardScaler
# loss :  [0.46001237630844116, 0.0033909252379089594]
# r2 스코어 :  0.6647555099350111 
#=============================================================================
# MaxAbsScaler
# loss :  17.01386070251465
# r2 스코어 :  0.7940637072715339
#=============================================================================
# RobustScaler
# loss :  17.2054500579834
# r2 스코어 :  0.7917446902703928

