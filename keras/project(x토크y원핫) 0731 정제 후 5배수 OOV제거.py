from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd


#1. 데이터

path = './_data/project_01/' # 경로 = .현재폴더 /하단
data_set = pd.read_csv(path + 'data_project_x5(220731).csv')

print(data_set)
print(data_set.shape) # (1515, 2)

print(data_set.columns) # Index(['분류', '글귀'], dtype='object')
# print(data_set.info())
# print(data_set.describe())


# 감정분류와 글귀를 나누어 각각 처리할 예정
# 감정분류를 x, 글귀를 y로 나누어 처리

#################################################################

#1-1. 첫번째 열 '분류' -- 감정분류

# 분류 데이터 토크나이징/패딩
x = data_set['분류']
token = Tokenizer()
token.fit_on_texts(x)
print(token.word_index)
print(len(token.word_index)) # 848

x1 = token.texts_to_sequences(x)
print(x1)

from keras.preprocessing.sequence import pad_sequences
pad_x1 = pad_sequences(x1, padding='pre', maxlen=13)

print(pad_x1)
print(pad_x1.shape) # (1515, 13)


#################################################################

#1-2. 두번째 열 '글귀' -- 글귀

# 글귀 데이터 레이블 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le = le.fit(data_set['글귀'])   # data_set['글귀']을 fit
data_set['글귀'] = le.transform(data_set['글귀'])   # data_set['글귀']에 따라 encoding
y = data_set['글귀']
print(y)
print(y.shape) # (1515,)

# 판다스 데이터프레임화
# col_name = range(1515)
# print(col_name) # range(0, 1515)

x = pd.DataFrame(pad_x1)
print(x)


# 다중분류 -- 원핫인코딩
from tensorflow.python.keras.utils.np_utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) # (1515, 302)

#################################################################

# 훈련/테스트 데이터 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66)
print(x_train.shape, x_test.shape) # (1060, 13) (455, 13)
print(y_train.shape, y_test.shape) # (1060, 302) (455, 302)

#################################################################

#2. 모델구성 -- Embedding, LSTM 사용
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding
model = Sequential()
model.add(Embedding(input_dim=848, output_dim=100, input_length=13)) 
model.add(LSTM(32, activation='tanh'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(302, activation='softmax'))
model.summary() 

#################################################################

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train, y_train, epochs=500, batch_size=100, validation_split=0.2)

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_ModelCheckPoint/project_01/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping =EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1, 
                             restore_best_weights=True) 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                      filepath= "".join([filepath, 'project_01_', date, '_', filename]))

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, 
                 validation_split=0.2, callbacks=[earlyStopping, mcp], verbose=1)



#################################################################

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

acc = model.evaluate(x, y)[1]  # [0] 넣으면 loss가 나옴
print('acc : ', acc)


# 예측에 사용할 x 값 (토크나이징/패딩)
x_predict = ['꾸준히 노력하면 이룰 수 있겠지? 끈기있는 태도로 임하고 싶은데 생각보다 쉽지 않다. 자신감을 가질 수 있을까?']
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


# 데이터 5배수


# + tanh

# epochs=1000, batch_size=100, patience=200
# output_dim=100 / activation='tanh'

# x_predict = ['꾸준히 노력하면 이룰 수 있겠지? 끈기있는 태도로 임하고 싶은데 생각보다 쉽지 않다. 자신감을 가질 수 있을까?']
# loss : [3.0745065212249756, 0.7802197933197021]
# acc :  0.8580858111381531
# 당신에게 들려주고 싶은 이야기는 ['어떻게 보이고 싶다고 해서 그렇게 보여지는 것이 아니다. 보는 사
# 람이 어떻게 보느냐에 따라 삼각산이 뾰족하게도 보이고 둥글게도 보이는 것이다.']
# project_01_0801_1418_0027-3.2044.hdf5

# loss : [2.508540391921997, 0.7296703457832336]
# acc :  0.8283828496932983
# 당신에게 들려주고 싶은 이야기는 ['사랑은 지배하는 것이 아니라 자유를 주는 것이다. 에리히 프롬']
# project_01_0801_1430_0023-2.6626.hdf5

# loss : [2.1535451412200928, 0.8373626470565796]
# acc :  0.8877887725830078
# 당신에게 들려주고 싶은 이야기는 ['이긴다고 생각하면 이긴다. 승리는 자신감을 가진 사람의 편이다. 
# 가토 마사오']
# project_01_0801_1504_0025-2.2970.hdf5

# loss : [2.466599225997925, 0.8109890222549438]
# acc :  0.8811880946159363
# 당신에게 들려주고 싶은 이야기는 ['행동의 가치는 그 행동을 끝까지 이루는 데 있다. 칭기스칸']
# project_01_0801_1511_0027-2.7427.hdf5

# loss : [2.451578378677368, 0.7890110015869141]
# acc :  0.8646864891052246
# 당신에게 들려주고 싶은 이야기는 ['토끼를 잡으려면 귀를 잡아야 하고, 고양이는 목덜미를 잡아야 한
# 다. 사람을 잡으려면 마음을 잡아야 한다.']
# project_01_0801_1517_0026-2.7094.hdf5

# loss : [2.5505387783050537, 0.7648351788520813]
# acc :  0.8448845148086548
# 당신에게 들려주고 싶은 이야기는 ['좋은 성과를 얻으려면 한 걸음 한 걸음이 힘차고 충실하지 않으면 
# 안된다. 단테']
# project_01_0801_1650_0024-2.7459.hdf5

