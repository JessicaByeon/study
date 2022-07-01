from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y) #y는 데이터 전처리를 할 필요가 없음.
print(x.shape, y.shape) #(442, 10) (442,) Number of Instances: 442, Number of Attributes 10

print(datasets.feature_names)
print(datasets.DESCR)

# [과제]
# # activation : sigmoid, relu, linear
# metrics 추가
# EarlyStopping 넣고
# 성능비교
# 느낀점 2줄 이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.9, shuffle=True, random_state=72)

#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='linear', input_dim=10))
model.add(Dense(7, activation='linear'))
model.add(Dense(10, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(12, activation='linear'))
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
hist = model.fit(x_train, y_train, epochs=200, batch_size=40, 
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
# loss :  1917.123291015625
# r2 스코어 :  0.6682922668838625
# [patience=100 일때]
# loss :  1998.3214111328125
# r2 스코어 :  0.6542430748632183



# 이 값을 이용해 그래프를 그려보자!

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='', c='red', label='loss') # 연속된 데이터는 엑스 빼고 와이만 써주면 됨. 순차적으로 진행.
plt.plot(hist.history['val_loss'], marker='', c='blue', label='val_loss')
plt.grid() # 모눈종이 형태로 볼 수 있도록 함
plt.title('이결바보')
plt.ylabel('loss')
plt.xlabel('epochs')
# plt.legend(loc='upper right') # 라벨값이 원하는 위치에 명시됨
plt.legend()
plt.show()




# 1/ validation 적용
# loss :  2343.607177734375
# r2 스코어 :  0.6450810741559115

# 2/ EarlyStopping 적용
# loss :  1998.3214111328125
# r2 스코어 :  0.6542430748632183

# 3/ activation 적용
# loss :  [1955.128173828125, 0.0]
# r2 스코어 :  0.6617165168421875

# 1/,2/,3/을 거치며 점점 손실도 줄고 r2 스코어도 증가하는 것을 발견!
# 최적화가 이루어진 것으로 보인다.
