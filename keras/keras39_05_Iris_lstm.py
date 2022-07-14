from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM

import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets['data']
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape) # (150, 4) (150,)

# 원핫인코딩은 모델구성 전 데이터 전처리에서 진행
print("y의 라벨값 : ", np.unique(y)) # y의 라벨값 :  [0 1 2] (총 3개가 있다는 것을 알 수 있음)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) #(150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)
#셔플을 잘 해주어야 데이터 분류에 오류가 없음
# print(y_train)
# print(y_test)

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


print(x_train.shape, x_test.shape) # (120, 4) (30, 4)
print(y_train.shape, y_test.shape) # (120, 3) (30, 3)
x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)
print(x_train.shape, x_test.shape) # (120, 4, 1) (30, 4, 1)


#2. 모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(4,1), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1)

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

# y_predict = model.predict(x_test)
# y_test = np.argmax(y_test, axis=1)
# y_predict = np.argmax(y_predict, axis=1)


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc) #acc 스코어 :


# LSTM
# loss :  0.13405779004096985
# accuracy :  0.9666666388511658
# ============= y_pred ==============
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 1 2 2 2 1 0 2 1 0 2 0 2 
# 2]
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 2 2 2 2 1 0 2 1 0 2 0 2 
# 2]
# acc 스코어 :  0.9666666666666667


# 1/ 이진분류 binary crossentropy 로 출력 
# train_size=0.75, patience=10, epochs=200, validation_split=0.2
# loss :  [-459849568.0, 0.31578946113586426] / 로스가 음수가 나왔다는 것은 지표가 잘못되었음을 의미
# acc 스코어 :  0.3157894736842105

# train_size=0.75, patience=50, epochs=500, validation_split=0.2
# loss :  [-132279496.0, 0.31578946113586426]
# acc 스코어 :  0.3157894736842105

# 다중분류로 취급되었어야하나 이진분류 기준으로 출력되었으므로 acc 스코어가 낮을 수밖에 없음.
# 다중분류로 취급하여 재계산 -- softmax 및 categorical crossentropy 사용

# 2/ 다중분류 softmax, categorical crossentropy 로 출력
# loss :  0.01792481169104576
# accuracy :  1.0
# acc 스코어 :  1.0

# loss :  0.068968765437603
# accuracy :  0.9666666388511658
# ============= y_pred ==============
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 1 2 2 2 1 0 2 1 0 2 0 2 2]
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 2 2 2 2 1 0 2 1 0 2 0 2 2]    
# acc 스코어 :  0.9666666666666667


#=============================================================================
# loss :  0.09458506107330322
# accuracy :  0.9333333373069763
# ============= y_pred ==============
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 1 2 2 2 1 0 2 1 0 2 0 2 1]
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 2 2 2 2 1 0 2 1 0 2 0 2 2]
# acc 스코어 :  0.9333333333333333
#=============================================================================
# MinMaxScaler
# loss :  0.10890697687864304
# accuracy :  0.9666666388511658
# ============= y_pred ==============
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 1 2 2 2 1 0 2 1 0 2 0 2 2]
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 2 2 2 2 1 0 2 1 0 2 0 2 2]
# acc 스코어 :  0.9666666666666667
#=============================================================================
# StandardScaler
# loss :  0.2063179910182953
# accuracy :  0.9666666388511658
# ============= y_pred ==============
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 1 2 2 2 1 0 2 1 0 2 0 2 2]
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 2 2 2 2 1 0 2 1 0 2 0 2 2]
# acc 스코어 :  0.9666666666666667
#=============================================================================
# MaxAbsScaler
# loss :  0.10415853559970856
# accuracy :  0.9666666388511658
# ============= y_pred ==============
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 1 2 2 2 1 0 2 1 0 2 0 2 2]
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 2 2 2 2 1 0 2 1 0 2 0 2 2]
# acc 스코어 :  0.9666666666666667
#=============================================================================
# RobustScaler
# loss :  0.1434842199087143
# accuracy :  0.9666666388511658
# ============= y_pred ==============
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 2 2 1 2 1 0 2 1 0 2 0 2 2]
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 2 2 2 2 1 0 2 1 0 2 0 2 2]
# acc 스코어 :  0.9666666666666667
