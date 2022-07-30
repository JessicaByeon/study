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
token = Tokenizer(oov_token="<OOV>")
token.fit_on_texts(x)
print(token.word_index)
# print(len(token.word_index)) # 257

x1 = token.texts_to_sequences(x)
print(x1)

from keras.preprocessing.sequence import pad_sequences
pad_x1 = pad_sequences(x1, padding='pre', maxlen=13)

print(pad_x1)
print(pad_x1.shape) # (220, 13)
# [[  0   0   0 ...  17  12   8]
#  [  0   0   0 ... 120  12   8]
#  [  0   0   0 ...  95  34  33]
#  ...
#  [  0   0   0 ... 255  53  65]
#  [  0   0   0 ...  11  13  38]
#  [  0   0   0 ...  41  22  24]]

x = pd.DataFrame(pad_x1)
print(x)
#      0   1   2   3   4   ...   8    9    10  11   12
# 0     0   0   0   0   0  ...  118  119   17  12    8     
# 1     0   0   0   0   0  ...   19   34  120  12    8     
# 2     0   0   0   0   0  ...   74   59   95  34   33     
# 3     0   0   0   0   0  ...    8    2   19  25  121     
# 4     0   0   0   0   0  ...   33   17   51   2   19     
# ..   ..  ..  ..  ..  ..  ...  ...  ...  ...  ..  ...     
# 215   0   0   0   0   0  ...   43   12    8   3    5     
# 216   0   0   0   0   0  ...   12    6    4  10    9     
# 217   0   0   0   0   0  ...   11    9  255  53   65     
# 218   0   0   0   0   0  ...   23  258   11  13   38     
# 219   0   0   0   0   0  ...    6   18   41  22   24 
# [220 rows x 13 columns]


#################################################################

#1-2. 두번째 열 '글귀' -- 글귀

# 글귀 데이터 레이블 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le = le.fit(data_set['글귀'])   # data_set['글귀']을 fit
data_set['글귀'] = le.transform(data_set['글귀'])   # data_set['글귀']에 따라 encoding
y = data_set['글귀']
print(y)
# 0      197
# 1       98
# 2        2
# 3      154
# 4      159
#       ...
# 215    205
# 216     40
# 217      5
# 218    177
# 219     35
# Name: 글귀, Length: 220, dtype: int32
print(y.shape) # (220,)

# 다중분류 -- 원핫인코딩
from tensorflow.python.keras.utils.np_utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) # (220, 220)

#################################################################

# 훈련/테스트 데이터 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66)
print(x_train.shape, x_test.shape) # (154, 13) (66, 13)
print(y_train.shape, y_test.shape) # (154, 220) (66, 220)

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
model.add(Dense(220, activation='softmax'))
model.summary() 

#################################################################

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train, y_train, epochs=100, batch_size=5000)

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 

hist = model.fit(x_train, y_train, epochs=100, batch_size=5000, 
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
print('당신에게 들려주고 싶은 이야기는', le.inverse_transform([y_predict[-1]]))

# 훈련 10
# x_predict = ['행복하고 싶다, 지금 당장!']
# loss : [5.410423278808594, 0.0]
# acc :  0.004545454401522875
# 당신에게 들려주고 싶은 이야기는 ['아무것도 하지 
# 않으면 아무 일도 일어나지 않는다. 기시미 이치로']

# 훈련 10
# x_predict = ['행복하고 싶다, 지금 당장!']
# loss : [5.408761501312256, 0.0]
# acc :  0.013636363670229912
# 당신에게 들려주고 싶은 이야기는 ['당신의
#  하루하루를 당신의 마지막 날이라고 생각 
# 하라. 호라티우스']

# 훈련 100
# x_predict = ['행복하고 싶다, 지금 당장!']
# loss : [95.2533950805664, 0.0]
# acc :  0.5363636612892151
# 당신에게 들려주고 싶은 이야기는 ['가장 
# 필요한 용기가 있다면, 그것은 나를 기꺼이
#  바꿀 용기. 이동영']