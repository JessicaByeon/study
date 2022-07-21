from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augument_size = 100 # 증폭

print(x_train[0].shape) # (28, 28)
print(x_train[0].reshape(28*28).shape) # (784,)
print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1).shape)
    # np.tile (x,y) x를 y만큼 반복
    # (100, 28, 28, 1)
    # np.tile 을 사용하여 증폭시키면 기존 데이터만 사용해서 증폭시켜주므로 과적합될 우려가 있음
print(np.zeros(augument_size))
print(np.zeros(augument_size).shape) # (100,)

# 여기까지는 모두 동일한 데이터를 반복시킨 데이터

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1), # x
    np.zeros(augument_size),                                                  # y 
    # flow from directory 는 경로가 들어가고 flow에는 x, y 데이터가 들어간다.
    batch_size=augument_size,
    shuffle=True,
)#.next()

#################### .next() 사용
# print(x_data) # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001EE5A401A60>
# print(x_data[0])
# print(x_data[0].shape) # (100, 28, 28, 1)
# print(x_data[1].shape) # (100,) 

# .next() 사용 시 첫 번째 []를 생략하고 진행


#################### .next() 미사용
print(x_data) # x, y가 묶여있는 배치단위 # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001EE5A401A60>
print(x_data[0])          # x와 y가 모두 포함
print(x_data[0][0].shape) # (100, 28, 28, 1) x값
print(x_data[0][1].shape) # (100,) y값


import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    # plt.imshow(x_data[0][i], cmap='gray') # .next() 사용
    plt.imshow(x_data[0][0][i], cmap='gray') # .next() 미사용
    
        # plt.imshow(x_data[i], cmap='gray')
plt.show()


# 이진분류일 때 가장 결과값이 좋을 때는 보통 2가지 데이터의 갯수가 거의 동일할 때
# 이진/다중분류일 때 데이터 한쪽/한가지의 갯수가 현저히 모자랄 때
# 부족한 데이터를 나머지 데이터의 숫자와 동일한 수치까지 맞추기 위해 데이터를 증폭시킬 때 많이 사용