# 넘파이 불러와서 모델링 할 것

import numpy as np
from keras.preprocessing.image import ImageDataGenerator # 이미지데이터 생성, 증폭

# 1. 데이터
# np.save('d:/study_data/_save/_npy/keras46_5_train_x_npy', arr=xy_train[0][0]) # train x 가 들어감
# np.save('d:/study_data/_save/_npy/keras46_5_train_y_npy', arr=xy_train[0][1]) # train y 가 들어감
# np.save('d:/study_data/_save/_npy/keras46_5_test_x_npy', arr=xy_test[0][0]) # test x 가 들어감
# np.save('d:/study_data/_save/_npy/keras46_5_test_y_npy', arr=xy_test[0][1]) # test y 가 들어감

x_train = np.load('d:/study_data/_save/_npy/keras47_1_cat_dog_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras47_1_cat_dog_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras47_1_cat_dog_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras47_1_cat_dog_test_y.npy')


print(x_train.shape, y_train.shape) # (500, 150, 150, 3) (500, 2)
print(x_test.shape, y_test.shape) # (500, 150, 150, 3) (500, 2)


#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=30, mode='min', verbose=1, 
                              restore_best_weights=True) 

hist = model.fit(x_train, y_train, epochs=100, 
                 validation_split=0.2, callbacks=[earlyStopping], verbose=1, batch_size=32)

# accuracy = hist.history['accuracy']
# val_accuracy = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# print('loss : ', loss[-1])
# print('val_loss : ', val_loss[-1])
# print('accuracy : ', accuracy[-1])
# print('val_accuracy : ', val_accuracy[-1])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)   

from sklearn.metrics import accuracy_score
y_predict = y_predict.round()
acc = accuracy_score(y_test,y_predict)

print('loss : ', loss)
print('accuracy : ', acc)
