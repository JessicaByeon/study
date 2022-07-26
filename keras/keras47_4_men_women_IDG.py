import numpy as np
from keras.preprocessing.image import ImageDataGenerator # 이미지데이터 생성, 증폭

train = ImageDataGenerator(
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

train = ImageDataGenerator(
    rescale=1./255
)

# 평가 데이터는 증폭시키지 않고 써야함. 확인하는/평가 데이터는 그대로 사용.
# train, test 데이터 조건을 지정한 후, 이미지를 불러와 엮어준다.

xydata = train.flow_from_directory(
    'D:\\study_data\\_data\\image\\archive\\data',
    target_size=(150,150),
    class_mode='binary',
    batch_size=500,
    shuffle=True,) # 경로 및 폴더 설정
# Found 3309 images belonging to 2 classes.


print(xydata)
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

# print(xydata[0][0],xydata[0][0].shape) # (500, 150, 150, 3)

# x = xydata[0][0]
# y = xydata[0][1]
abc = train.flow_from_directory(
    'D:\\study_data\\_data\\image\\archive\\abc',
    target_size=(150,150),
    class_mode='binary',
    batch_size=1,
    shuffle=True,) # 경로 및 폴더 설정

np.save('d:/study_data/_save/_npy/keras47_4_men_women_train_x.npy', arr=xydata[0][0])
np.save('d:/study_data/_save/_npy/keras47_4_men_women_train_y.npy', arr=xydata[0][1])
np.save('d:/study_data/_save/_npy/keras47_4_men_women_train_z.npy', arr=xydata[0][0])





