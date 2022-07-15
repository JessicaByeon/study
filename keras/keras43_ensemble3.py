#1. 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301, 401)]) # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]) # 원유, 돈육, 밀
x3_datasets = np.array([range(100, 200), range(1301, 1401)]) # 우리반 아이큐, 우리반 키
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

##### 실습 #####

print(x1.shape, x2.shape, x3.shape) # (100, 2) (100, 3) (100, 2)

y1 = np.array(range(2001, 2101)) # (100, ) / 금리
y2 = np.array(range(201, 301)) # (100, )

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, \
y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, x3, y1, y2, train_size=0.7, shuffle=True, random_state=66)
print(x1_train.shape, x1_test.shape) # (70, 2) (30, 2)
print(x2_train.shape, x2_test.shape) # (70, 3) (30, 3)
print(x3_train.shape, x3_test.shape) # (70, 2) (30, 2)
print(y1_train.shape, y1_test.shape) # (70, ) (30,)
print(y2_train.shape, y2_test.shape) # (70, ) (30,)


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

#2-3. 모델3
input3 = Input(shape=(2,))
dense21 = Dense(16, activation='relu', name='jb21')(input3)
dense22 = Dense(32, activation='relu', name='jb22')(dense21)
dense23 = Dense(32, activation='relu', name='jb23')(dense22)
dense24 = Dense(32, activation='relu', name='jb24')(dense23)
output3 = Dense(10, activation='relu', name='out_jb3')(dense24)


# Concatenate (Class / 함수 -- 2가지를 사용 가능)
from tensorflow.python.keras.layers import concatenate, Concatenate # 소문자 함수, 대문자 클래스, 사슬처럼 엮다, 리스트의 append 개념처럼 합치다.
# merge1 = concatenate([output1, output2, output3], name='mg1') # concat으로 엮어서 하나의 dense layer로 만들어줌
merge1 = Concatenate()([output1, output2, output3]) # name을 사용할 수 없는지?
merge2 = Dense(16, activation='relu', name='mg2')(merge1)
merge3 = Dense(16, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

#2-4. output모델1
output41 = Dense(64, activation='relu')(last_output)
output42 = Dense(64, activation='relu')(output41)
last_output2 = Dense(1)(output42)

#2-5. output모델2
output51 = Dense(64, activation='relu')(last_output)
output52 = Dense(64, activation='relu')(output51)
output53 = Dense(64, activation='relu')(output52)
last_output3 = Dense(1)(output53)

# 모델에 대한 정의
model = Model(inputs=[input1, input2, input3], outputs=[last_output2, last_output3])


# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=500, batch_size=100)

#4. 평가, 예측
# loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
loss1 = model.evaluate([x1_test, x2_test, x3_test], y1_test) 
loss2 = model.evaluate([x1_test, x2_test, x3_test], y2_test)

print('loss1 :', loss1)
print('loss2 :', loss2)

from sklearn.metrics import r2_score
y1_pred, y2_pred = model.predict([x1_test, x2_test, x3_test])
print(y1_pred)
print(y2_pred)

r2_1 = r2_score(y1_test, y1_pred)
r2_2 = r2_score(y2_test, y2_pred)

print('r2 스코어1 : ', r2_1)
print('r2 스코어2 : ', r2_2)

# loss1 : [3212295.75, 171.01206970214844, 3212124.75]
# loss2 : [3225544.75, 3224887.0, 657.7318115234375]
# r2 스코어1 :  0.9856154174207548
# r2 스코어2 :  0.17801505461095468


# 로스는 상대적인 것
# 두 개를 더한 값인 전체 로스를 다른 훈련한 것과 비교하기 때문에 상관이 없음?
