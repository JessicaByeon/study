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
    'd:/_data/image/brain/train/',
    target_size=(200,200), # 크기 조절 / 크기가 다른 이미지들을 해당 사이즈로 리사이징
    batch_size=5, # 크게 줘도 에러는 나지 않는다. 자동으로 최대값에 맞춰 진행
    class_mode='binary', # 흑백이므로 분류가 0,1 두가지
    color_mode='grayscale',
    shuffle=True,
) # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory( # 디렉토리(폴더)에서 가져온 것을 위와 같은 조건으로 생성해서 xy_train에 집어넣겠다.
    'd:/_data/image/brain/test/',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
) # Found 120 images belonging to 2 classes.

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001B4F74B52E0> 

# from sklearn.datasets import load_boston
# datasets = load_boston()
# print(datasets)

print(xy_train[0]) # x와 y값이 같이 포함되어있고, y가 5개 포함되어있다. batch_size = 5
# 총 160개 데이터가 배치 5개 단위로 잘려있고, 5개씩 총 32개의 단위로 구성되어 있음
print(xy_train[31]) # 마지막 배치 / 0번째는 x, 1번째는 y
# print(xy_train[31][0].shape) # (5, 150, 150, 3) 3 : 흑백도 컬러 데이터이므로 기본적으로 '컬러' 데이터로 인식
print(xy_train[31][0].shape) # x값 쉐입 (5, 150, 150, 1) 1 : 위쪽에서 컬러 조절 color_mode='grayscale' 넣음
print(xy_train[31][1]) # y값[1. 0. 1. 0. 1.]
# print(xy_train[31][2]) # 0과 1만 존재하므로 2는 없음 : 에러 출력

print(xy_train[31][0].shape, xy_train[31][1].shape)
# (5, 200, 200, 1) (5,)

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

# loss :  0.6854770183563232
# val_loss :  0.7038587927818298
# accuracy :  0.539393961429596
# val_accuracy :  0.5

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