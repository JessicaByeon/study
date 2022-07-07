<<<<<<< HEAD
# [과제]
# 3가지 원핫인코딩 방식을 비교할것!

# OneHotEncoding?
# 인덱스에 1값을 부여하고, 나머지 인덱스에 0을 부여하는 표현방식
# pandas get_dummies, tensorflow.keras to_categorical, sklearn OneHotEncoder 함수사용



# 미세한 차이를 정리하시오.
# 아래에 3가지 방식을 써서 fetch_covtype 에 대한 출력과정 및 출력값을 표시해둠

import numpy as np
from sklearn.datasets import fetch_covtype, load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import pandas as pd

import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) 
# [1 2 3 4 5 6 7] dim 7 softmax (1797,54)으로 원핫인코딩 평가지표 accuracy
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
# print(datasets.DESCR)
# print(datasets.feature_names)



#==============================================================================
#1/ pandas의 get_dummies
y = pd.get_dummies(y)

# loss :  0.6646304726600647
# accuracy :  0.7235957980155945
# ============= y_pred ==============
# tf.Tensor([1 1 0 ... 2 1 1], shape=(116203,), dtype=int64)
# tf.Tensor([1 1 0 ... 5 1 1], shape=(116203,), dtype=int64)       
# acc 스코어 :  0.7235957763568927



#==============================================================================
#2/ tensorflow의 to_categorical
# array 배열에서 0부터 빈 곳을 모두 채워줌
# 해당 과정에서도 실제 데이터는 1부터 시작하지만, 0부터 시작하게되므로 0이 추가된 상태가 됨
# [1, 2, 3, 4, 5, 6, 7] --> [0, 1, 2, 3, 4, 5, 6, 7]로 맨 앞 0이 채워짐

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) # (581012, 8)

# loss :  0.7133055925369263
# accuracy :  0.6944915652275085
# ============= y_pred ==============
# [2 2 1 ... 3 2 1]
# [2 2 1 ... 6 2 2]
# acc 스코어 :  0.6944915363630887



#==============================================================================
#3/ sklearn의 OneHotEncoder

from sklearn.preprocessing import OneHotEncoder
y = np.array(y).reshape(-1,1)
print(y.shape) # (581012, 8)

ohe = OneHotEncoder()
ohe.fit(y)

y_class = ohe.transform(y).toarray()
print(y_class.shape) # (581012, 7)
# 해당 기능을 통해 y값을 클래스 수에 맞는 열로 늘리는 원핫 인코딩 처리를 한다.
# 1개의 컬럼으로 [0,1,2] 였던 값을 ([1,0,0],[0,1,0],[0,0,1]과 같은 shape로 만들어줌)

# loss :  0.6646304726600647
# accuracy :  0.7235957980155945
# ============= y_pred ==============
# [1 1 0 ... 5 1 1]
# [1 1 0 ... 2 1 1]
# acc 스코어 :  0.7235957763568927



#==============================================================================
x_train, x_test, y_train, y_test = train_test_split(x, y_class,
        train_size=0.8, shuffle=True, random_state=68)



#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='linear', input_dim=54))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)
model.fit(x_train, y_train, epochs=10, batch_size=32, 
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
y_test = np.argmax(y_test,axis=1) # pandas get dummies --> tf.argmax 적용, sklearn OneHotEncoding --> np.argmax 적용
print(y_test)
y_predict = np.argmax(y_predict,axis=1) # y_test와 y_predict의 shape가 일치해야함
print(y_predict)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)
=======
# [과제]
# 3가지 원핫인코딩 방식을 비교할것!

# OneHotEncoding?
# 인덱스에 1값을 부여하고, 나머지 인덱스에 0을 부여하는 표현방식
# pandas get_dummies, tensorflow.keras to_categorical, sklearn OneHotEncoder 함수사용



# 미세한 차이를 정리하시오.
# 아래에 3가지 방식을 써서 fetch_covtype 에 대한 출력과정 및 출력값을 표시해둠

import numpy as np
from sklearn.datasets import fetch_covtype, load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import pandas as pd

import tensorflow as tf
tf.random.set_seed(66)
# 웨이트의 난수

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True)) 
# [1 2 3 4 5 6 7] dim 7 softmax (1797,54)으로 원핫인코딩 평가지표 accuracy
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
# print(datasets.DESCR)
# print(datasets.feature_names)



#==============================================================================
#1/ pandas의 get_dummies
y = pd.get_dummies(y)

# loss :  0.6646304726600647
# accuracy :  0.7235957980155945
# ============= y_pred ==============
# tf.Tensor([1 1 0 ... 2 1 1], shape=(116203,), dtype=int64)
# tf.Tensor([1 1 0 ... 5 1 1], shape=(116203,), dtype=int64)       
# acc 스코어 :  0.7235957763568927



#==============================================================================
#2/ tensorflow의 to_categorical
# array 배열에서 0부터 빈 곳을 모두 채워줌
# 해당 과정에서도 실제 데이터는 1부터 시작하지만, 0부터 시작하게되므로 0이 추가된 상태가 됨
# [1, 2, 3, 4, 5, 6, 7] --> [0, 1, 2, 3, 4, 5, 6, 7]로 맨 앞 0이 채워짐

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) # (581012, 8)

# loss :  0.7133055925369263
# accuracy :  0.6944915652275085
# ============= y_pred ==============
# [2 2 1 ... 3 2 1]
# [2 2 1 ... 6 2 2]
# acc 스코어 :  0.6944915363630887



#==============================================================================
#3/ sklearn의 OneHotEncoder

from sklearn.preprocessing import OneHotEncoder
y = np.array(y).reshape(-1,1)
print(y.shape) # (581012, 8)

ohe = OneHotEncoder()
ohe.fit(y)

y_class = ohe.transform(y).toarray()
print(y_class.shape) # (581012, 7)
# 해당 기능을 통해 y값을 클래스 수에 맞는 열로 늘리는 원핫 인코딩 처리를 한다.
# 1개의 컬럼으로 [0,1,2] 였던 값을 ([1,0,0],[0,1,0],[0,0,1]과 같은 shape로 만들어줌)

# loss :  0.6646304726600647
# accuracy :  0.7235957980155945
# ============= y_pred ==============
# [1 1 0 ... 5 1 1]
# [1 1 0 ... 2 1 1]
# acc 스코어 :  0.7235957763568927



#==============================================================================
x_train, x_test, y_train, y_test = train_test_split(x, y_class,
        train_size=0.8, shuffle=True, random_state=68)



#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='linear', input_dim=54))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)
model.fit(x_train, y_train, epochs=10, batch_size=32, 
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
y_test = np.argmax(y_test,axis=1) # pandas get dummies --> tf.argmax 적용, sklearn OneHotEncoding --> np.argmax 적용
print(y_test)
y_predict = np.argmax(y_predict,axis=1) # y_test와 y_predict의 shape가 일치해야함
print(y_predict)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)
>>>>>>> 425a270641f2121145cc0df2169ab516031dcf7e
