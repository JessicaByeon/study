import numpy as np
from keras.preprocessing.image import ImageDataGenerator # 이미지데이터 생성, 증폭

# 1. 데이터
# np.save('d:/study_data/_save/_npy/keras46_5_train_x_npy', arr=xy_train[0][0]) # train x 가 들어감
# np.save('d:/study_data/_save/_npy/keras46_5_train_y_npy', arr=xy_train[0][1]) # train y 가 들어감
# np.save('d:/study_data/_save/_npy/keras46_5_test_x_npy', arr=xy_test[0][0]) # test x 가 들어감
# np.save('d:/study_data/_save/_npy/keras46_5_test_y_npy', arr=xy_test[0][1]) # test y 가 들어감

x_train = np.load('d:/study_data/_save/_npy/keras46_5_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras46_5_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras46_5_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras46_5_test_y.npy')


print(x_train)
print(x_train.shape) # (160, 150, 150, 1)
print(y_train.shape) # (160,)
print(x_test.shape) # (120, 200, 200, 1)
print(y_test.shape) # (120,)





'''
# 자료형 확인
print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'> iterator : 반복자
print(type(xy_train[0])) # <class 'tuple'> x 하나 튜플, y 하나 튜플
print(type(xy_train[0][0])) # <class 'numpy.ndarray'> np x, np y가 배치단위로 묶여있음
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

# 현재 5, 200, 200, 1 짜리 데이터가 32 덩어리



#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(200, 200, 1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(xy_train[0][0], xy_train[0][1]) # 배치를 최대로 잡으면 이것도 가능
hist = model.fit_generator(xy_train, epochs=30, steps_per_epoch=33, # 전체 데이터를 batch size로 나눈 값 160/5 = 32
                    validation_data=xy_test,
                    validation_steps=4) # fit generator에 통으로 넣어주면 됨

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_accuracy : ', val_accuracy[-1])

# loss :  9.401800525665749e-06
# val_loss :  0.12895289063453674
# accuracy :  1.0
# val_accuracy :  0.949999988079071

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss') # 연속된 데이터는 엑스 빼고 와이만 써주면 됨. 순차적으로 진행.
plt.plot(hist.history['val_loss'], marker='.', c='purple', label='val_loss')
plt.plot(hist.history['accuracy'], marker='.', c='blue', label='accuracy') # 연속된 데이터는 엑스 빼고 와이만 써주면 됨. 순차적으로 진행.
plt.plot(hist.history['val_accuracy'], marker='.', c='skyblue', label='val_accuracy')
plt.grid() # 모눈종이 형태로 볼 수 있도록 함
plt.title('결과값')
plt.ylabel('loss/accuracy')
plt.xlabel('epochs')
plt.legend(loc='upper right') # 라벨값이 원하는 위치에 명시됨
plt.legend()
plt.show()
'''