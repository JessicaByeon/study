<<<<<<< HEAD
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten # 이미지 작업은 2D

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


# model.add(Conv2D(filters=10, kernel_size=(2,2), # 10 다음 레이어로 전달해주는 아웃풋 노드의 갯수, kernel size 이미지를 자르는 규격
#                  input_shape=(5, 5, 1))) # (batch_size, rows, columns, channels)   # 출력 : (N, 4, 4, 10)
# model.add(Conv2D(7, (2,2), activation='relu')) # filter = 7, kernel size = (2,2) # 출력 : (N, 3, 3, 7)

# model.add(Flatten()) # (N, 63) 3*3*7
# model.add(Dense(32, activation='relu')) # flatten이 없을경우, (N, 3, 3, 32) 로 출력
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.summary()
# (kernel_size * channels + bias) * filters = summary Param 갯수 (CNN 모델)

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 4, 4, 10)          50
# =================================================================
# Total params: 50
# Trainable params: 50
# Non-trainable params: 0

# output shape
# (5-2)+1/1 = 4
# (5-2)+1/1 = 4
# 출력채널이 10,
# 4, 4, 10

# param #
# (2*2 + 1(bias)) * 10(output) = 50


# Model: "sequential"
# _________________________________________________________________     
# Layer (type)                 Output Shape              Param #        
# =================================================================
# conv2d (Conv2D)              (None, 4, 4, 10)          50
# _________________________________________________________________     
# conv2d_1 (Conv2D)            (None, 3, 3, 7)           287
# =================================================================     
# Total params: 337
# Trainable params: 337
# Non-trainable params: 0


mmodel.add(Conv2D(filters=10, kernel_size=(3,3), # 10 다음 레이어로 전달해주는 아웃풋 노드의 갯수, kernel size 이미지를 자르는 규격
                 input_shape=(8, 8, 1))) # (batch_size, rows, columns, channels)   # 출력 : (N, 6, 6, 10)
model.add(Conv2D(7, (2,2), activation='relu')) # filter = 7, kernel size = (2,2) # 출력 : (N, 5, 5, 7)

model.add(Flatten()) # (N, 175) 5*5*7
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
# (kernel_size * channels + bias) * filters = summary Param 갯수 (CNN 모델)
# ((3*3*1)+1)*10 = 10*10 = 100 
=======
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten # 이미지 작업은 2D

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


# model.add(Conv2D(filters=10, kernel_size=(2,2), # 10 다음 레이어로 전달해주는 아웃풋 노드의 갯수, kernel size 이미지를 자르는 규격
#                  input_shape=(5, 5, 1))) # (batch_size, rows, columns, channels)   # 출력 : (N, 4, 4, 10)
# model.add(Conv2D(7, (2,2), activation='relu')) # filter = 7, kernel size = (2,2) # 출력 : (N, 3, 3, 7)

# model.add(Flatten()) # (N, 63) 3*3*7
# model.add(Dense(32, activation='relu')) # flatten이 없을경우, (N, 3, 3, 32) 로 출력
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# model.summary()
# (kernel_size * channels + bias) * filters = summary Param 갯수 (CNN 모델)

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 4, 4, 10)          50
# =================================================================
# Total params: 50
# Trainable params: 50
# Non-trainable params: 0

# output shape
# (5-2)+1/1 = 4
# (5-2)+1/1 = 4
# 출력채널이 10,
# 4, 4, 10

# param #
# (2*2 + 1(bias)) * 10(output) = 50


# Model: "sequential"
# _________________________________________________________________     
# Layer (type)                 Output Shape              Param #        
# =================================================================
# conv2d (Conv2D)              (None, 4, 4, 10)          50
# _________________________________________________________________     
# conv2d_1 (Conv2D)            (None, 3, 3, 7)           287
# =================================================================     
# Total params: 337
# Trainable params: 337
# Non-trainable params: 0


mmodel.add(Conv2D(filters=10, kernel_size=(3,3), # 10 다음 레이어로 전달해주는 아웃풋 노드의 갯수, kernel size 이미지를 자르는 규격
                 input_shape=(8, 8, 1))) # (batch_size, rows, columns, channels)   # 출력 : (N, 6, 6, 10)
model.add(Conv2D(7, (2,2), activation='relu')) # filter = 7, kernel size = (2,2) # 출력 : (N, 5, 5, 7)

model.add(Flatten()) # (N, 175) 5*5*7
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
# (kernel_size * channels + bias) * filters = summary Param 갯수 (CNN 모델)
# ((3*3*1)+1)*10 = 10*10 = 100 
>>>>>>> 0032b7bc9af5be1bd054bebd127a94da7509d68b
# ((2*2*10)+1)*7 = 41*7 = 287