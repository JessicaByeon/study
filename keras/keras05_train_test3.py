import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# train과 test를 섞어서 7:3으로 찾을 수 있는 방법을 찾아라
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x, y, 
    test_size=0.3,     #test size / train size 둘 중 하나만 써도 됨
    train_size=0.7, 
    # shuffle=True,    #default가 true이므로 없어도 됨
    random_state=66
    ) #***랜덤난수(랜덤값고정을위해), 난수표 참고, 난수값을 66을 씀

print(x_train) #[2 7 6 3 4 8 5]
print(x_test) #[ 1  9 10] 
print(y_train) #[2 7 6 3 4 8 5]
print(y_test) #[ 1  9 10] 


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(3))
model.add(Dense(1)) #hidden layer가 하나라도 딥러닝

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
result = model.predict([11])
print('[11]의 예측값 :', result)

# loss : 4.263256414560601e-14
# [11]의 예측값 : [[11.]]