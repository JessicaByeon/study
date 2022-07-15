#1. 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301, 401)]) # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]) # 원유, 돈육, 밀
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)

print(x1.shape, x2.shape) # (100, 2) (100, 3)

y = np.array(range(2001, 2101)) # (100, ) / 금리

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, shuffle=True, random_state=66)
print(x1_train.shape, x1_test.shape) # (70, 2) (30, 2)
print(x2_train.shape, x2_test.shape) # (70, 3) (30, 3)
print(y_train.shape, y_test.shape) # (70, ) (30,)

#2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(16, activation='relu', name='jb1')(input1)
dense2 = Dense(32, activation='relu', name='jb2')(dense1)
dense3 = Dense(32, activation='relu', name='jb3')(dense2)
output1 = Dense(10, activation='relu', name='out_jb1')(dense3)

#2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(16, activation='relu', name='jb11')(input2)
dense12 = Dense(32, activation='relu', name='jb12')(dense11)
dense13 = Dense(32, activation='relu', name='jb13')(dense12)
dense14 = Dense(32, activation='relu', name='jb14')(dense13)
output2 = Dense(10, activation='relu', name='out_jb2')(dense14)

from tensorflow.python.keras.layers import concatenate, Concatenate # 소문자 함수, 대문자 클래스, 사슬처럼 엮다, 리스트의 append 개념처럼 합치다.
merge1 = concatenate([output1, output2], name='mg1') # concat으로 엮어서 하나의 dense layer로 만들어줌
merge2 = Dense(16, activation='relu', name='mg2')(merge1)
merge3 = Dense(16, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # 회귀모델이므로, 딱 떨어지지 않는 결과 값 예측 mse
model.fit([x1_train, x2_train], y_train, epochs=500, batch_size=100)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss :', loss)

from sklearn.metrics import r2_score
y_pred = model.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_pred)
print('r2 스코어 : ', r2)

# loss : 3.272362232208252
# r2 스코어 :  0.9962575149699069




