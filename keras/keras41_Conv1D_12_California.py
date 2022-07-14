# 아래 모델에 대해 3가지 비교

# 스케일링 하기 전
# MinMaxScaler
# StandardScaler

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten
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

#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='linear', input_dim=8))
model.add(Dense(9, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(9, activation='linear'))
model.add(Dense(1, activation='linear'))

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

'''
print('------------------------------')
print(hist) # <tensorflow.python.keras.callbacks.History object at 0x00000219A7310F40>
print('------------------------------')
print(hist.history) 
print('------------------------------')
print(hist.history['loss']) #키밸류 상의 loss는 이름이기 때문에 ''를 넣어줌
print('------------------------------')
print(hist.history['val_loss']) #키밸류 상의 val_loss는 이름이기 때문에 ''를 넣어줌
'''

# print("걸린시간 : ", end_time)

# 그래프 그리기 전에 r2
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

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


'''
# 이 값을 이용해 그래프를 그려보자!

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # 연속된 데이터는 엑스 빼고 와이만 써주면 됨. 순차적으로 진행.
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() # 모눈종이 형태로 볼 수 있도록 함
plt.title('이결바보')
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right') # 라벨값이 원하는 위치에 명시됨
plt.legend()
plt.show()
'''



# 1/ validation 적용
# loss :  0.6926966309547424
# r2 스코어 :  0.5031103996989461

# 2/ EarlyStopping 적용
# loss :  0.6468518972396851
# r2 스코어 :  0.5285920019123878

# 3/ activation 적용
# loss :  [0.5073163509368896, 0.0033909252379089594]
# r2 스코어 :  0.630281735559099

# 1/,2/,3/을 거치며 점점 손실도 줄고 r2 스코어도 증가하는 것을 발견!
# 최적화가 이루어진 것으로 보인다.

