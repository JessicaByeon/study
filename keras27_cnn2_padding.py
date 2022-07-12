from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D # 이미지 작업은 2D

model = Sequential()
# model.add(Dense(units=10, input_shape=(3,1))) # (batch_size, input_dim) 형태  #input_shape=(10, 10, 3)))
# model.summary()
# (input_dim + bias) * units = summary Param 갯수(Dense모델)

# Model: "sequential"
# _________________________________________________________________     
# Layer (type)                 Output Shape              Param #        
# =================================================================     
# dense (Dense)                (None, 3, 10)             20
# =================================================================     
# Total params: 20
# Trainable params: 20
# Non-trainable params: 0

# model.add(Conv2D(filters=64, kernel_size=(3,3), # 10 다음 레이어로 전달해주는 아웃풋 노드의 갯수, kernel size 이미지를 자르는 규격
#                  padding='same', # 원래 shape를 그대로 유지하여 다음 레이어로 보내주고 싶을 때 주로 사용!
#                  input_shape=(28, 28, 1))) # (batch_size, rows, columns, channels)   # 출력 : (N, 28, 28, 64)
# model.add(MaxPooling2D()) # (N, 14, 14, 64)
# model.add(Conv2D(32, (2,2), 
#                  padding='valid', # 디폴트
#                  activation='relu')) # filter = 7, kernel size = (2,2) # 출력 : (N, 13, 13, 32)

# model.add(Flatten()) # (N, 175) 5*5*7
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.summary()
# (kernel_size * channels + bias) * filters = summary Param 갯수 (CNN 모델)
# ((3*3*1)+1)*10 = 10*10 = 100 
# ((2*2*10)+1)*7 = 41*7 = 287


# MaxPooling
# input_shape을 홀수로 해보면 어떤 값이 나오는지 확인해보자.
model.add(Conv2D(filters=64, kernel_size=(3,3), # 10 다음 레이어로 전달해주는 아웃풋 노드의 갯수, kernel size 이미지를 자르는 규격
                 padding='same', # 원래 shape를 그대로 유지하여 다음 레이어로 보내주고 싶을 때 주로 사용!
                 input_shape=(13, 13, 1))) # (batch_size, rows, columns, channels)   # 출력 : (N, 13, 13, 64)
model.add(MaxPooling2D()) # (N, 6, 6, 64)
model.add(Conv2D(32, (2,2), 
                 padding='valid', # 디폴트
                 activation='relu')) # filter = 7, kernel size = (2,2) # 출력 : (N, 5, 5, 32)

model.add(Flatten()) # (N, 175) 5*5*7
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) # 10개의 분류모델
model.summary()
# (kernel_size * channels + bias) * filters = summary Param 갯수 (CNN 모델)
# ((3*3*1)+1)*10 = 10*10 = 100 
# ((2*2*10)+1)*7 = 41*7 = 287