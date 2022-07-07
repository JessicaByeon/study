<<<<<<< HEAD
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) :
    print('쥐피유 돈다')
    aaa = 'gpu'
else:
    print('쥐피유 안도라')
    aaa = 'cpu'



#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR) # (569, 30)
# print(datasets.feature_names)

x = datasets['data'] # datasets.data 로도 쓸 수 있음
y = datasets['target']
print(x.shape, y.shape) # (569, 30) (569,)

# print(x)
# print(y) # 0, 1이 569개

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.9, shuffle=True, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(500, activation='linear', input_dim=30))
model.add(Dense(400, activation='sigmoid'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True) 

# loss :  0.0955263003706932

# earlyStopping 보통 변수는 앞글자 소문자
# 모니터 val_loss 대신 loss도 가능

import time
start_time = time.time() # 현재 시간 출력
hist = model.fit(x_train, y_train, epochs=100, batch_size=20, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

end_time = time.time() - start_time # 걸린 시간
print(aaa, '걸린시간 : ', end_time)

# cpu 걸린시간 :  7.133010625839233
# gpu 걸린시간 :  14.835393190383911 / 작은 데이터는 gpu가 더 걸릴 수도 있음


# print('------------------------------')
# print(hist) # <tensorflow.python.keras.callbacks.History object at 0x00000219A7310F40>
# print('------------------------------')
# print(hist.history) 
# print('------------------------------')
# print(hist.history['loss']) #키밸류 상의 loss는 이름이기 때문에 ''를 넣어줌
# print('------------------------------')
# print(hist.history['val_loss']) #키밸류 상의 val_loss는 이름이기 때문에 ''를 넣어줌

# print("걸린시간 : ", end_time)


# 그래프 그리기 전에 r2/acc
y_predict = model.predict(x_test)
y_predict[(y_predict<0.5)] = 0  
y_predict[(y_predict>=0.5)] = 1 
print(y_predict)
print(y_predict)

#### [과제 1.] accuracy score 완성

from sklearn.metrics import r2_score, accuracy_score
# # r2 = r2_score(y_test, y_predict)
# # print('r2 스코어 : ', r2)
# # r2 스코어 :  0.5852116219896948

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc) #acc 스코어 :  0.8947368421052632
print(y_predict)


# 이 값을 이용해 그래프를 그려보자!
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # 연속된 데이터는 엑스 빼고 와이만 써주면 됨. 순차적으로 진행.
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() # 모눈종이 형태로 볼 수 있도록 함
plt.title('breast cancer')
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right') # 라벨값이 원하는 위치에 명시됨
plt.legend()
=======
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) :
    print('쥐피유 돈다')
    aaa = 'gpu'
else:
    print('쥐피유 안도라')
    aaa = 'cpu'



#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR) # (569, 30)
# print(datasets.feature_names)

x = datasets['data'] # datasets.data 로도 쓸 수 있음
y = datasets['target']
print(x.shape, y.shape) # (569, 30) (569,)

# print(x)
# print(y) # 0, 1이 569개

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.9, shuffle=True, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(500, activation='linear', input_dim=30))
model.add(Dense(400, activation='sigmoid'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(400, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True) 

# loss :  0.0955263003706932

# earlyStopping 보통 변수는 앞글자 소문자
# 모니터 val_loss 대신 loss도 가능

import time
start_time = time.time() # 현재 시간 출력
hist = model.fit(x_train, y_train, epochs=100, batch_size=20, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

end_time = time.time() - start_time # 걸린 시간
print(aaa, '걸린시간 : ', end_time)

# cpu 걸린시간 :  7.133010625839233
# gpu 걸린시간 :  14.835393190383911 / 작은 데이터는 gpu가 더 걸릴 수도 있음


# print('------------------------------')
# print(hist) # <tensorflow.python.keras.callbacks.History object at 0x00000219A7310F40>
# print('------------------------------')
# print(hist.history) 
# print('------------------------------')
# print(hist.history['loss']) #키밸류 상의 loss는 이름이기 때문에 ''를 넣어줌
# print('------------------------------')
# print(hist.history['val_loss']) #키밸류 상의 val_loss는 이름이기 때문에 ''를 넣어줌

# print("걸린시간 : ", end_time)


# 그래프 그리기 전에 r2/acc
y_predict = model.predict(x_test)
y_predict[(y_predict<0.5)] = 0  
y_predict[(y_predict>=0.5)] = 1 
print(y_predict)
print(y_predict)

#### [과제 1.] accuracy score 완성

from sklearn.metrics import r2_score, accuracy_score
# # r2 = r2_score(y_test, y_predict)
# # print('r2 스코어 : ', r2)
# # r2 스코어 :  0.5852116219896948

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc) #acc 스코어 :  0.8947368421052632
print(y_predict)


# 이 값을 이용해 그래프를 그려보자!
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # 연속된 데이터는 엑스 빼고 와이만 써주면 됨. 순차적으로 진행.
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid() # 모눈종이 형태로 볼 수 있도록 함
plt.title('breast cancer')
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right') # 라벨값이 원하는 위치에 명시됨
plt.legend()
>>>>>>> 425a270641f2121145cc0df2169ab516031dcf7e
plt.show()