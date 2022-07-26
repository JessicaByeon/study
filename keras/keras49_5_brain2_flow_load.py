# 넘파이에서 불러와서 모델 구성
# 성능 비교


import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, LSTM, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd

x_train = np.load('d:/study_data/_save/_npy/keras49_5_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_5_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_5_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_5_test_y.npy')

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

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 

# model.fit(xy_train[0][0], xy_train[0][1]) # 배치를 최대로 잡으면 이것도 가능
hist = model.fit(x_train, y_train, epochs=550, batch_size=5000, 
                validation_split=0.2,
                callbacks=[earlyStopping], # 최저값을 체크해 반환해줌
                verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 

y_predict = model.predict(x_test)
print(y_test.shape, y_predict.shape) # (10000, 1) (10000, 10)

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
# print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
# print('val_accuracy : ', val_accuracy[-1])


# loss :  0.12991932034492493
# accuracy :  0.9607843160629272


# [k46_2] 
# loss :  0.6854770183563232
# val_loss :  0.7038587927818298
# accuracy :  0.539393961429596
# val_accuracy :  0.5

'''
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