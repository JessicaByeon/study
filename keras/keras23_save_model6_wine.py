# 함수형 모델구성으로 변경

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input

import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)
print(np.unique(y, return_counts=True)) # [0 1 2] (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print(datasets.DESCR)
# print(datasets.feature_names)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) #(178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)

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
# model = Sequential()
# model.add(Dense(5, activation='linear', input_dim=13))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(3, activation='softmax'))

# input1 = Input(shape=(13,)) # 먼저 input layer를 명시해줌
# dense1 = Dense(5, activation='linear')(input1)
# dense2 = Dense(10, activation='relu')(dense1)
# dense3 = Dense(10, activation='relu')(dense2)
# dense4 = Dense(10, activation='linear')(dense3)
# output1 = Dense(3, activation='softmax')(dense4)
# model = Model(inputs=input1, outputs=output1)


# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['accuracy'])

# from tensorflow.python.keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, 
#                               restore_best_weights=True)

# hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, 
#                 validation_split=0.2,
#                 callbacks=[earlyStopping],
#                 verbose=1)

# model.save("./_save/keras23_save_model6_wine.h5")
model = load_model("./_save/keras23_save_model6_wine.h5")

#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)

# 아래와 같이 표기도 가능!
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('accuracy : ', result[1])

# print("============= y_test[:5] ==============")
# print(y_test[:5])
# print("============= y_pred ==============")
# y_predict = model.predict(x_test[:5])
# print(y_predict)
print("============= y_pred ==============")

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)

# epochs=500, batch_size=20,
# loss :  0.15111111104488373
# accuracy :  0.9444444179534912
# ============= y_pred ==============
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 1 2 1 0 1 1 1 1 1 2 0 
# 2 1 2 2]
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 1 2 1 0 1 1 1 1 2 2 0 
# 2 1 2 1]
# acc 스코어 :  0.9444444444444444



#=============================================================================
# loss :  0.192671537399292
# accuracy :  0.9166666865348816
# ============= y_pred ==============
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 1 2 1 0 1 0 1 1 2 0 0 2 1 2 2]
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 1 2 1 0 1 1 1 1 2 2 0 2 1 2 1]    
# acc 스코어 :  0.9166666666666666
#=============================================================================
# MinMaxScaler
# loss :  0.0666208416223526
# accuracy :  0.9722222089767456
# ============= y_pred ==============
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 1 2 1 0 1 1 1 1 2 1 0 2 1 2 1]
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 1 2 1 0 1 1 1 1 2 2 0 2 1 2 1]    
# acc 스코어 :  0.9722222222222222
#=============================================================================
# StandardScaler
# loss :  0.20279686152935028
# accuracy :  0.9166666865348816
# ============= y_pred ==============
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 0 2 1 0 1 0 1 1 2 1 0 2 1 2 1]
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 1 2 1 0 1 1 1 1 2 2 0 2 1 2 1]    
# acc 스코어 :  0.9166666666666666
#=============================================================================
# MaxAbsScaler
# loss :  0.2356511801481247
# accuracy :  0.9444444179534912
# ============= y_pred ==============
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 0 2 1 0 1 1 1 1 2 2 0 2 0 2 1]
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 1 2 1 0 1 1 1 1 2 2 0 2 1 2 1]    
# acc 스코어 :  0.9444444444444444
#=============================================================================
# RobustScaler
# loss :  0.22286632657051086
# accuracy :  0.9166666865348816
# ============= y_pred ==============
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 0 2 1 0 1 0 1 1 2 1 0 2 1 2 1]
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 1 2 1 0 1 1 1 1 2 2 0 2 1 2 1]    
# acc 스코어 :  0.9166666666666666

# 함수형 모델 =================================================================
# MinMaxScaler
# loss :  0.06662081182003021
# accuracy :  0.9722222089767456
# ============= y_pred ==============
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 1 2 1 0 1 1 1 1 2 1 0 2 1 2 1]
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 1 2 1 0 1 1 1 1 2 2 0 2 1 2 1]
# acc 스코어 :  0.9722222222222222