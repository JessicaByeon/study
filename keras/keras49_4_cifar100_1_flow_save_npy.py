# 증폭해서 npy에 저장

from tensorflow.keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

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

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

augument_size = 50000
batch_size = 64

randidx = np.random.randint(x_train.shape[0], size=augument_size)
# randint 랜덤하게 정수값을 넣는다.

x_augumented = x_train[randidx].copy() # 뽑은 4만개의 변수를 augumented 변수에 넣겠다. .copy() 원본을 건들지 않고 다른 공간에 저장
y_augumented = y_train[randidx].copy() # 뽑은 4만개의 변수를 augumented 변수에 넣겠다. .copy() 원본을 건들지 않고 다른 공간에 저장print(x_augumented.shape) # (40000, 28, 28)
print(y_augumented.shape) # (50000,)
print(x_augumented.shape) # (50000, 32, 32, 3)
# 기존 데이터를 그대로 복사한 데이터

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)

x_augumented = x_augumented.reshape(x_augumented.shape[0], 
                                    x_augumented.shape[1], 
                                    x_augumented.shape[2], 3)

x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False).next()[0]
# print(x_augumented[0][1])

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))
print(x_train.shape, y_train.shape) # (100000, 32, 32, 3) (100000, 1)

xy_train = train_datagen.flow(x_train, y_train, 
                               batch_size=batch_size,
                               shuffle=False)

# print(xy_train[0][0])
# print(xy_train[0][0].shape)

# print(xy_train[0][0].shape) #(100000, 28, 28, 1)
# print(xy_train[0][1].shape) #(100000,)

np.save('d:/study_data/_save/_npy/keras49_4_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras49_4_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/keras49_4_test_x.npy', arr=x_test)
np.save('d:/study_data/_save/_npy/keras49_4_test_y.npy', arr=y_test)
