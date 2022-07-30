from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd


#1. 데이터

path = './_data/project_01/' # 경로 = .현재폴더 /하단
data_set = pd.read_csv(path + 'data_project_(220729).csv')

print(data_set)
print(data_set.shape) # (220, 2)

print(data_set.columns) # Index(['분류', '글귀'], dtype='object')
# print(data_set.info())
# print(data_set.describe())


# 감정분류와 글귀를 나누어 각각 처리할 예정
# 감정분류를 x, 글귀를 y로 나누어 처리

#################################################################

#1-1. 첫번째 열 '분류' -- 감정분류

# 분류 데이터 토크나이징/패딩
x = data_set['분류']
y = data_set['글귀']
token = Tokenizer(oov_token="<OOV>")
token.fit_on_texts(x+' '+y)
print(token.word_index)
print(len(token.word_index)) # 1878

x1 = token.texts_to_sequences(x)
print(x1)
print('==============================')
y1 = token.texts_to_sequences(y)
print(y1)

from keras.preprocessing.sequence import pad_sequences
pad_x1 = pad_sequences(x1, padding='pre', maxlen=13)
print(pad_x1)
print(pad_x1.shape) # (220, 13)

pad_y1 = pad_sequences(y1, padding='pre', maxlen=33)
print(pad_y1)
print(pad_y1.shape) # (220, 33)


# 판다스 데이터프레임화
x = pd.DataFrame(pad_x1)
print(x)
y = pd.DataFrame(pad_y1)
print(y)


'''
#################################################################

#1-2. 두번째 열 '글귀' -- 글귀

# 분류 데이터 토크나이징/패딩
y = data_set['글귀']
token.fit_on_texts(y)
print(token.word_index)
print(len(token.word_index)) # 257

x1 = token.texts_to_sequences(x)
print(x1)

from keras.preprocessing.sequence import pad_sequences
pad_x1 = pad_sequences(x1, padding='pre', maxlen=13)

print(pad_x1)
print(pad_x1.shape) # (220, 13)

'''
#################################################################

# 훈련/테스트 데이터 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66)
print(x_train.shape, x_test.shape) # (154, 13) (66, 13)
print(y_train.shape, y_test.shape) # (154, 33) (66, 33)

#################################################################

#2. 모델구성 -- Embedding, LSTM 사용
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding
model = Sequential()
model.add(Embedding(input_dim=257, output_dim=220, input_length=13)) 
model.add(LSTM(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(33, activation='softmax'))
model.summary() 

#################################################################

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train, y_train, epochs=100, batch_size=5000)

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 

hist = model.fit(x_train, y_train, epochs=10, batch_size=5000, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1)

#################################################################

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

acc = model.evaluate(x, y)[1]  # [0] 넣으면 loss가 나옴
print('acc : ', acc)


# 예측에 사용할 x 값 (토크나이징/패딩)
x_predict = ['행복하고 싶다, 지금 당장!']
# token.fit_on_texts(x_predict)
# print(token.word_index)

x_predict1 = token.texts_to_sequences(x_predict)
# print(x_predict1)

from keras.preprocessing.sequence import pad_sequences
pad_x_predict1 = pad_sequences(x_predict1, padding='pre', maxlen=13)
# print(pad_x_predict1)


y_predict = model.predict(pad_x_predict1)
y_predict = np.argmax(y_predict, axis= 1)
print('당신에게 들려주고 싶은 이야기는', y_predict[-1])
