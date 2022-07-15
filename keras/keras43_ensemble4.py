#1. 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301, 401)]) # 삼성전자 종가, 하이닉스 종가
x1 = np.transpose(x1_datasets)

##### 실습 #####

print(x1.shape) # (100, 2)

y1 = np.array(range(2001, 2101)) # (100, ) / 금리
y2 = np.array(range(201, 301)) # (100, )

from sklearn.model_selection import train_test_split

x1_train, x1_test, \
y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, y1, y2, train_size=0.7, shuffle=True, random_state=66)
print(x1_train.shape, x1_test.shape) # (70, 2) (30, 2)
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

# # Concatenate (Class / 함수 -- 2가지를 사용 가능)
# from tensorflow.python.keras.layers import concatenate, Concatenate # 소문자 함수, 대문자 클래스, 사슬처럼 엮다, 리스트의 append 개념처럼 합치다.
# merge1 = concatenate([output1, output2, output3], name='mg1') # concat으로 엮어서 하나의 dense layer로 만들어줌
# # merge1 = Concatenate()([output1, output2, output3]) # name을 사용할 수 없는지?
# merge2 = Dense(16, activation='relu', name='mg2')(merge1)
# merge3 = Dense(16, name='mg3')(merge2)
# last_output = Dense(1, name='last')(merge3)

#2-4. output모델1
output41 = Dense(64, activation='relu')(output1)
output42 = Dense(64, activation='relu')(output41)
last_output2 = Dense(1)(output42)

#2-5. output모델2
output51 = Dense(64, activation='relu')(output1)
output52 = Dense(64, activation='relu')(output51)
output53 = Dense(64, activation='relu')(output52)
last_output3 = Dense(1)(output53)

# 모델에 대한 정의
model = Model(inputs=input1, outputs=[last_output2, last_output3])


# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x1_train, [y1_train, y2_train], epochs=500, batch_size=100)

#4. 평가, 예측
# loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
loss1 = model.evaluate(x1_test, y1_test) 
loss2 = model.evaluate(x1_test, y2_test)

print('loss1 :', loss1)
print('loss2 :', loss2)

from sklearn.metrics import r2_score
y1_pred, y2_pred = model.predict(x1_test)
print(y1_pred)
print(y2_pred)

r2_1 = r2_score(y1_test, y1_pred)
r2_2 = r2_score(y2_test, y2_pred)

print('r2 스코어1 : ', r2_1)
print('r2 스코어2 : ', r2_2)

# loss1 : [3219172.5, 16.44355583190918, 3219156.0]
# loss2 : [3234704.25, 3234386.75, 317.48846435546875]
# r2 스코어1 :  0.9811940872721739
# r2 스코어2 :  0.6368996867339825


# 로스는 상대적인 것
# 두 개를 더한 값인 전체 로스를 다른 훈련한 것과 비교하기 때문에 상관이 없음?
