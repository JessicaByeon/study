# 넘파이에서 불러와서 모델 구성
# 성능 비교


import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, LSTM, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd

x_train = np.load('d:/study_data/_save/_npy/keras49_1_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_1_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_1_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_1_test_y.npy')

# print(np.unique(y_train, return_counts=True))
# # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# #  array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))

# # reshape 할 때 모든 개체를 곱한 값은 동일해야한다.
# # 모양은 바꿀 수 있다. 다만 데이터 순서만 바뀌지 않으면 됨

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), # 64 다음 레이어로 전달해주는 아웃풋 노드의 갯수, kernel size 이미지를 자르는 규격
                 padding='same', # 원래 shape를 그대로 유지하여 다음 레이어로 보내주고 싶을 때 주로 사용!
                 input_shape=(28, 28, 1))) # (batch_size, rows, columns, channels)   # 출력 : (N, 28, 28, 64)
model.add(MaxPooling2D()) # (N, 14, 14, 64)
model.add(Conv2D(32, (2,2), 
                 padding='valid', # 디폴트
                 activation='relu')) # filter = 32, kernel size = (2,2) # 출력 : (N, 13, 13, 32)
model.add(Conv2D(32, (2,2), 
                 padding='valid', # 디폴트
                 activation='relu')) # filter = 32, kernel size = (2,2) # 출력 : (N, 12, 12, 32)
model.add(Flatten()) # (N, 4608) 12*12*32
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
# model.summary()


#3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['accuracy'])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 

hist = model.fit(x_train, y_train, epochs=550, batch_size=5000, 
                validation_split=0.2,
                callbacks=[earlyStopping], # 최저값을 체크해 반환해줌
                verbose=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 

y_predict = model.predict(x_test)
print(y_test.shape, y_predict.shape)

acc = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('accuracy : ', acc[-1])


# loss :  6.362973363138735e-05
# accuracy :  1.0


# [keras48_6]
# loss :  [0.4935600161552429, 0.8953999876976013]   
# acc스코어 :  0.8954

# [keras39_16]
# LSTM
# loss :  [4.631586074829102, 0.009999999776482582]

# CNN
# loss :  [0.3275989294052124, 0.8998000025749207]
# acc스코어 :  0.8998