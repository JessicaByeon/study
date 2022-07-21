from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

augument_size = 40000 # 4만장 늘리기! 총 10만장이 됨
randidx = np.random.randint(x_train.shape[0], size=augument_size)
# randint 랜덤하게 정수값을 넣는다.
print(x_train.shape[0]) # 60000
# x_train.shape -> (60000, 28, 28) 인데 이 shape의 0번째 이므로 60000
# randint(60000, 40000) -> 0~59999 까지의 숫자 중에서 40000개를 뽑아내겠다.
print(randidx) # [15912 37918   147 ... 50606 19887 29041]
print(np.min(randidx), np.max(randidx)) # 1 59998
print(type(randidx)) # <class 'numpy.ndarray'> 기본적으로 리스트 형태

x_augumented = x_train[randidx].copy() # 뽑은 4만개의 변수를 augumented 변수에 넣겠다. .copy() 원본을 건들지 않고 다른 공간에 저장
y_augumented = y_train[randidx].copy() # 뽑은 4만개의 변수를 augumented 변수에 넣겠다. .copy() 원본을 건들지 않고 다른 공간에 저장print(x_augumented.shape) # (40000, 28, 28)
print(y_augumented.shape) # (40000,)
# 기존 데이터를 그대로 복사한 데이터

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_augumented = x_augumented.reshape(x_augumented.shape[0], 
                                    x_augumented.shape[1], 
                                    x_augumented.shape[2], 1)

x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False).next()[0] # 0번째 x를 넣겠다.
print(x_augumented)
print(x_augumented.shape) # (40000, 28, 28, 1)

# y는 현재 있는 그대로 써도 되므로... y_augumented 는 따로 안해줘도 됨.

# x_train 과 x_augumented 를 concat 해주자.
x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)




#### [실습] 모델구성 ####
# 성능비교, 증폭 전 후 비교




import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, LSTM, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
#  array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))

# reshape 할 때 모든 개체를 곱한 값은 동일해야한다.
# 모양은 바꿀 수 있다. 다만 데이터 순서만 바뀌지 않으면 됨

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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
model.compile(loss='categorical_crossentropy', optimizer='adam',
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
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, accuracy_score
y_predict = np.argmax(y_predict, axis= 1)
y_predict = to_categorical(y_predict)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)


# loss :  [0.4935600161552429, 0.8953999876976013]   
# acc스코어 :  0.8954



# [keras39_16]
# LSTM
# loss :  [4.631586074829102, 0.009999999776482582]

# CNN
# loss :  [0.3275989294052124, 0.8998000025749207]
# acc스코어 :  0.8998
