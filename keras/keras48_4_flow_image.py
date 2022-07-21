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
y_augumented = y_train[randidx].copy() # 뽑은 4만개의 변수를 augumented 변수에 넣겠다. .copy() 원본을 건들지 않고 다른 공간에 저장
print(x_augumented.shape) # (40000, 28, 28)
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

# [실습]
# x_augumented 10개와 x_train 10개를 비교하는 이미지 출력할 것! 

# train 에서 10개 고르고
# agumented 10개 동일한 이미지로 고르고
# 비교
# 이미지 출력 subplot

augument_size=10
x_train = x_train[1:11]
print(x_train.shape) # (10, 28, 28, 1)
randidx2 = np.random.randint(x_train.shape[0], size=augument_size)
x_augumented2 = x_train[randidx2].copy()
print(x_augumented2.shape) # (10, 28, 28, 1)



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

#original image와 transformation된 image 시각화해서 보기
plt.subplot(1, 2, 1)
plt.title('original')
plt.imshow(np.squeeze(image), 'gray')
plt.subplot(1, 2, 2)
plt.title('Transforms Image')
plt.imshow(np.squeeze(image_result), 'gray')
plt.show()
'''