from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd


#1. 데이터

path = './_data/project_01/' # 경로 = .현재폴더 /하단
data_set = pd.read_csv(path + 'data_project_(220731).csv')

print(data_set)
print(data_set.shape) # (303, 2)

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
# print(len(token.word_index)) # 744

x1 = token.texts_to_sequences(x)
print(x1)

from keras.preprocessing.sequence import pad_sequences
pad_x1 = pad_sequences(x1, padding='pre', maxlen=13)

print(pad_x1)
print(pad_x1.shape) # (303, 13)


#################################################################

#1-2. 두번째 열 '글귀' -- 글귀

# 글귀 데이터 레이블 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le = le.fit(data_set['글귀'])   # data_set['글귀']을 fit
data_set['글귀'] = le.transform(data_set['글귀'])   # data_set['글귀']에 따라 encoding
y = data_set['글귀']
print(y)
print(y.shape) # (303,)

# 판다스 데이터프레임화
# col_name = range(303)
# print(col_name) # range(0, 303)

x = pd.DataFrame(pad_x1)
print(x)


# 다중분류 -- 원핫인코딩
from tensorflow.python.keras.utils.np_utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) # (303, 302)

#################################################################

# 훈련/테스트 데이터 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66)
print(x_train.shape, x_test.shape) # (212, 13) (91, 13)
print(y_train.shape, y_test.shape) # (212, 302) (91, 302)

#################################################################

#2. 모델구성 -- Embedding, LSTM 사용
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding
model = Sequential()
model.add(Embedding(input_dim=744, output_dim=302, input_length=13)) 
model.add(LSTM(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(302, activation='softmax'))
model.summary() 

#################################################################

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train, y_train, epochs=100, batch_size=5000)

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 

hist = model.fit(x_train, y_train, epochs=500, batch_size=5000, 
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
# x_predict = ['노력 의지 인내']
# loss : [5.423771381378174, 0.0]
# acc :  0.00909090880304575
# 당신에게 들려주고 싶은 이야기는 ['망설이
# 면 두려움만 커진다.']

# 훈련 10
# x_predict = ['노력 의지 인내']
# loss : [5.413045406341553, 0.0]
# acc :  0.022727273404598236
# 당신에게 들려주고 싶은 이야기는 ['새로운
#  미래를 원한다면 그 미래에 걸맞게 행동하
# 라. 아무리 두려워도 그냥 시작하라.'] 

# 데이터 추가(220731) 후

# 훈련 10
# x_predict = ['행복하고 싶다, 지금 당장!']
# loss : [5.730127811431885, 0.0]
# acc :  0.013201320543885231
# acc :  0.013201320543885231
# 당신에게 들려주고 싶은 이야기는 ['짊어지는 무게가 아닌, 짊어지는 방법이 중요해요.']

# 훈련 10
# x_predict = ['행복하고 싶다, 지금 당장!']
# loss : [112.69136047363281, 0.0]
# acc :  0.5214521288871765
# 당신에게 들려주고 싶은 이야기는 ['말하지 말고 행동하세요. 말하지 말고 보여주세요. 약속하지 말고 증
# 명하세요.']