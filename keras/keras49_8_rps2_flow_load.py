# 넘파이에서 불러와서 모델 구성
# 성능 비교


import numpy as np
from keras.preprocessing.image import ImageDataGenerator # 이미지데이터 생성, 증폭

# 1. 데이터
# np.save('d:/study_data/_save/_npy/keras46_5_train_x_npy', arr=xy_train[0][0]) # train x 가 들어감
# np.save('d:/study_data/_save/_npy/keras46_5_train_y_npy', arr=xy_train[0][1]) # train y 가 들어감
# np.save('d:/study_data/_save/_npy/keras46_5_test_x_npy', arr=xy_test[0][0]) # test x 가 들어감
# np.save('d:/study_data/_save/_npy/keras46_5_test_y_npy', arr=xy_test[0][1]) # test y 가 들어감

x = np.load('d:/study_data/_save/_npy/keras49_8_x.npy')
y = np.load('d:/study_data/_save/_npy/keras49_8_y.npy')

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.3, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape) # (350, 150, 150, 3) (350,)
print(x_test.shape, y_test.shape) # (150, 150, 150, 3) (150,)


#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
acc = accuracy_score(y_test, y_predict)

print('loss : ', loss)
print('accuracy : ', acc)

# loss :  [0.37581324577331543, 0.9599999785423279]
# accuracy :  0.96

# [k47_3]
# loss :  [0.19784928858280182, 0.9666666388511658]
# accuracy :  0.96