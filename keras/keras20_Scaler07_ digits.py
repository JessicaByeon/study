# 아래 모델에 대해 3가지 비교

# 스케일링 하기 전
# MinMaxScaler
# StandardScaler

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,) 8*8 이미지가 1797개 있다는..
print(np.unique(y, return_counts=True)) 
# [0 1 2 3 4 5 6 7 8 9] dim 10 softmax (1797,10)으로 원핫인코딩 평가지표 accuracy
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
# print(datasets.DESCR)
# print(datasets.feature_names)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) #(1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=68)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train) # 수치로 변환해주는 걸 x_train에 집어넣자.
# x_test = scaler.transform(x_test) 
# print(np.min(x_train))
# print(np.max(x_train))
# print(np.min(x_test))
# print(np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='linear', input_dim=64))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=500, batch_size=100, 
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

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)

'''
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.image[2])
plot.show()
'''

# loss :  0.4753378927707672
# accuracy :  0.8888888955116272


#=============================================================================
# loss :  0.3976026773452759
# accuracy :  0.9083333611488342
# acc 스코어 :  0.9083333333333333
#=============================================================================
# MinMaxScaler
# loss :  0.39715245366096497
# accuracy :  0.9027777910232544
# acc 스코어 :  0.9027777777777778
#=============================================================================
# StandardScaler
# loss :  0.3944777250289917
# accuracy :  0.8888888955116272
# acc 스코어 :  0.8888888888888888
#=============================================================================
# MaxAbsScaler
# loss :  0.39715245366096497
# accuracy :  0.9027777910232544
# acc 스코어 :  0.9027777777777778
#=============================================================================
# RobustScaler
# loss :  0.5001651644706726
# accuracy :  0.8416666388511658
# acc 스코어 :  0.8416666666666667
