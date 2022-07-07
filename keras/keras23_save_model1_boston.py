# 함수형 모델구성으로 변경

from pyexpat import model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sqlalchemy import false
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input

datasets = load_boston()
x = datasets.data
y = datasets['target']

# print(np.min(x)) # 0.0
# print(np.max(x)) # 711.0 --- 711을 최대값인 1로 잡고 계산
# x = (x - np.min(x)) / (np.max(x) - np.min(x))
# print(x[:10])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # 수치로 변환해주는 걸 x_train에 집어넣자.
x_test = scaler.transform(x_test) 
# print(np.min(x_train)) # 0.0
# print(np.max(x_train)) # 1.0000000000000002
# print(np.min(x_test)) # -0.06141956477526944
# print(np.max(x_test)) # 1.1478180091225068

#2. 모델구성
# model = Sequential()
# model.add(Dense(5, input_dim=13))
# model.add(Dense(9))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(1))

# input1 = Input(shape=(13,)) # 먼저 input layer를 명시해줌
# dense1 = Dense(5)(input1)
# dense2 = Dense(9, activation='relu')(dense1)
# dense3 = Dense(10, activation='sigmoid')(dense2)
# dense4 = Dense(10, activation='relu')(dense3)
# output1 = Dense(1)(dense4)
# model = Model(inputs=input1, outputs=output1)


# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# # start_time = time.time() #현재 시간 출력 1656032967.2581124
# # print(start_time) #1656032967.2581124

# hist = model.fit(x_train, y_train, epochs=500, batch_size=20, 
#                 validation_split=0.2,
#                 verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음

# end_time = time.time() - start_time #걸린 시간

# model.save("./_save/keras23_save_model1_boston.h5")
model = load_model("./_save/keras23_save_model1_boston.h5")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

#=============================================================================
# loss :  21.426353454589844
# r2 스코어 :  0.7406547626963109
#=============================================================================
# MinMaxScaler
# loss :  17.020431518554688
# r2 스코어 :  0.793984149564475
#=============================================================================
# StandardScaler
# loss :  16.821285247802734
# r2 스코어 :  0.7963946205369385
#=============================================================================
# MaxAbsScaler
# loss :  17.01386070251465
# r2 스코어 :  0.7940637072715339
#=============================================================================
# RobustScaler
# loss :  17.2054500579834
# r2 스코어 :  0.7917446902703928

# 스케일링을 했을 때 더 결과값이 좋음. 
# 해당 모델에서는 StandardScaler가 가장 좋은 결과값을 가짐.

# 함수형 모델 =================================================================
# StandardScaler
# loss :  10.037687301635742
# r2 스코어 :  0.8785034955062345




'''
print('------------------------------')
print(hist) # <tensorflow.python.keras.callbacks.History object at 0x00000219A7310F40>
print('------------------------------')
print(hist.history) 
print('------------------------------')
print(hist.history['loss']) #키밸류 상의 loss는 이름이기 때문에 ''를 넣어줌
print('------------------------------')
print(hist.history['val_loss']) #키밸류 상의 val_loss는 이름이기 때문에 ''를 넣어줌

# print("걸린시간 : ", end_time)


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


# {'loss': [1521.7059326171875, 130.54188537597656, 91.58596801757812, 82.65372467041016, 
# 72.5574951171875, 70.0708236694336, 66.46566009521484, 70.04518127441406, 63.4539794921875, 63.71456527709961, 61.172828674316406], 'val_loss': [158.5045623779297, 101.32865905761719, 85.67953491210938, 82.06331634521484, 70.68658447265625, 87.28343200683594, 101.1255874633789, 73.15115356445312, 63.8593635559082, 113.73959350585938, 72.89293670654297]}
# 딕셔너리{} 키밸류 형태로 loss와 val_loss를 반환해줌

# val 없이 반환하면 로스만 반환
# val 적용하여 반환하면 로스, val_loss 두가지 반환

# 히스토리 안에 반환값 - loss. val_loss
# 로스에서 최저값을 찾을 수 있음, 그 지점이 최적의 Weight
