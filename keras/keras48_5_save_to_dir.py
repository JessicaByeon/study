# 증폭된 데이터를 파일로 저장

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

augument_size = 20 # 4만장 늘리기! 총 10만장이 됨
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

import time
start_time = time.time()
print('시작!')
x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  save_to_dir='d:/study_data/_temp',
                                  shuffle=False).next()[0] # 0번째 x를 넣겠다.

end_time = time.time() - start_time

# print('걸린시간 : ', round(end_time, 3), '초')
# # 시작!
# # 걸린시간 :  0.036 초

print(augument_size, '개 증폭에 걸린시간 : ', round(end_time, 3), '초')
# 시작!
# 20 개 증폭에 걸린시간 :  0.051 초


# print(x_augumented)
# print(x_augumented.shape) # (40000, 28, 28, 1)

# # y는 현재 있는 그대로 써도 되므로... y_augumented 는 따로 안해줘도 됨.

# # x_train 과 x_augumented 를 concat 해주자.
# x_train = np.concatenate((x_train, x_augumented))
# y_train = np.concatenate((y_train, y_augumented))

# print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)