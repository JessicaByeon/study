import numpy as np
from keras.preprocessing.image import ImageDataGenerator # 이미지데이터 생성, 증폭

train_datagen = ImageDataGenerator(
    rescale=1./255, # 225로 나눈다 / 최소값 0(블랙) ~ 최대값 255(화이트) 255로 나눈다는 것은 스케일링을 하겠다라는 의미(MinMax)
    # horizontal_flip=True, # 반전 여부 / 예
    # vertical_flip=True,
    # width_shift_range=0.1, # 수평이동 10%
    # height_shift_range=0.1, # 상하이동
    # rotation_range=5, # 회전
    # zoom_range=1.2, # 확대
    # shear_range=0.7,
    # fill_mode='nearest'
) # 주석처리한 부분 --> 이미지의 변환없이 실행가능

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# 평가 데이터는 증폭시키지 않고 써야함. 확인하는/평가 데이터는 그대로 사용.
# train, test 데이터 조건을 지정한 후, 이미지를 불러와 엮어준다.

xy_train = train_datagen.flow_from_directory( # 디렉토리(폴더)에서 가져온 것을 위와 같은 조건으로 생성해서 xy_train에 집어넣겠다.
    'd:/study_data/_data/image/horse-or-human/horse-or-human',
    target_size=(150,150), # 크기 조절 / 크기가 다른 이미지들을 해당 사이즈로 리사이징
    batch_size=500, # 크게 줘도 에러는 나지 않는다. 자동으로 최대값에 맞춰 진행
    class_mode='binary',
    shuffle=True,
) # Found 1027 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory( # 디렉토리(폴더)에서 가져온 것을 위와 같은 조건으로 생성해서 xy_train에 집어넣겠다.
    'd:/study_data/_data/image/horse-or-human/horse-or-human',
    target_size=(150,150),
    batch_size=500,
    class_mode='binary',
    shuffle=True,
) # Found 1027 images belonging to 2 classes.


print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001B4F74B52E0> 

# from sklearn.datasets import load_boston
# datasets = load_boston()
# print(datasets)

# print(xy_train[0]) # x와 y값이 같이 포함되어있고, y가 5개 포함되어있다. batch_size = 5
# # 총 160개 데이터가 배치 5개 단위로 잘려있고, 5개씩 총 32개의 단위로 구성되어 있음
# print(xy_train[31]) # 마지막 배치 / 0번째는 x, 1번째는 y
# # print(xy_train[31][0].shape) # (5, 150, 150, 3) 3 : 흑백도 컬러 데이터이므로 기본적으로 '컬러' 데이터로 인식
# print(xy_train[31][0].shape) # x값 쉐입 (5, 150, 150, 1) 1 : 위쪽에서 컬러 조절 color_mode='grayscale' 넣음
# print(xy_train[31][1]) # y값[1. 0. 1. 0. 1.]
# # print(xy_train[31][2]) # 0과 1만 존재하므로 2는 없음 : 에러 출력

print(xy_train[0][0].shape, xy_train[0][1].shape) # (500, 150, 150, 3) (500,)
print(xy_test[0][0].shape, xy_test[0][1].shape) # (500, 150, 150, 3) (500,)

np.save('d:/study_data/_save/_npy/keras47_2_horse-or-human_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras47_2_horse-or-human_y.npy', arr=xy_train[0][1])

