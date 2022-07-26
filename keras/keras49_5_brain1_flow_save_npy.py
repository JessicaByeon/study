# 증폭해서 npy에 저장

import numpy as np
from keras.preprocessing.image import ImageDataGenerator # 이미지데이터 생성, 증폭

train_datagen = ImageDataGenerator(
    rescale=1./255, # 225로 나눈다 / 최소값 0(블랙) ~ 최대값 255(화이트) 255로 나눈다는 것은 스케일링을 하겠다라는 의미(MinMax)
    horizontal_flip=True, # 반전 여부 / 예
    vertical_flip=True,
    width_shift_range=0.1, # 수평이동 10%
    height_shift_range=0.1, # 상하이동
    rotation_range=5, # 회전
    zoom_range=1.2, # 확대
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# 평가 데이터는 증폭시키지 않고 써야함. 확인하는/평가 데이터는 그대로 사용.
# train, test 데이터 조건을 지정한 후, 이미지를 불러와 엮어준다.

xy_train = train_datagen.flow_from_directory( # 디렉토리(폴더)에서 가져온 것을 위와 같은 조건으로 생성해서 xy_train에 집어넣겠다.
    'd:/study_data/_data/image/brain/train/',
    target_size=(200,200), # 크기 조절 / 크기가 다른 이미지들을 해당 사이즈로 리사이징
    batch_size=160, # 크게 줘도 에러는 나지 않는다. 자동으로 최대값에 맞춰 진행
    class_mode='binary', # 흑백이므로 분류가 0,1 두가지
    color_mode='grayscale',
    shuffle=True,
) # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory( # 디렉토리(폴더)에서 가져온 것을 위와 같은 조건으로 생성해서 xy_train에 집어넣겠다.
    'd:/study_data/_data/image/brain/test/',
    target_size=(200,200),
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
) # Found 120 images belonging to 2 classes.

print(xy_train[0][0], xy_train[0][1].shape) # (160, 200, 200, 1) (120, 200, 200, 1)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, x_test.shape) # (160,)
print(y_train.shape, y_test.shape) # (120,)

augument_size = 100
batch_size = 64

randidx = np.random.randint(x_train.shape[0], size=augument_size)
# randint 랜덤하게 정수값을 넣는다.

x_augumented = x_train[randidx].copy() # 뽑은 4만개의 변수를 augumented 변수에 넣겠다. .copy() 원본을 건들지 않고 다른 공간에 저장
y_augumented = y_train[randidx].copy() # 뽑은 4만개의 변수를 augumented 변수에 넣겠다. .copy() 원본을 건들지 않고 다른 공간에 저장print(x_augumented.shape) # (40000, 28, 28)
print(x_augumented.shape) # (100, 200, 200, 1)
print(y_augumented.shape) # (100,)
# 기존 데이터를 그대로 복사한 데이터

# x_train = x_train.reshape(100, 200, 200, 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)

# x_augumented = x_augumented.reshape(x_augumented.shape[0], 
#                                     x_augumented.shape[1], 
#                                     x_augumented.shape[2], 3)

x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False).next()[0]
# print(x_augumented[0][1])

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))
print(x_train.shape, y_train.shape) # (260, 200, 200, 1) (260,)

xy_train = train_datagen.flow(x_train, y_train, 
                               batch_size=batch_size,
                               shuffle=False)

print(xy_train[0][0].shape) # (64, 200, 200, 1)
print(xy_train[0][1].shape) # (64,)

np.save('d:/study_data/_save/_npy/keras49_5_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras49_5_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/keras49_5_test_x.npy', arr=x_test)
np.save('d:/study_data/_save/_npy/keras49_5_test_y.npy', arr=y_test)
