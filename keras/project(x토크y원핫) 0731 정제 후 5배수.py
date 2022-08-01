from cmath import tanh
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
token = Tokenizer(oov_token="<OOV>")
token.fit_on_texts(x)
print(token.word_index)
print(len(token.word_index)) # 849

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
model.add(Embedding(input_dim=849, output_dim=100, input_length=13)) 
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

# epochs=1000, batch_size=100, patience=200
# output_dim=302
# x_predict = ['꾸준히 노력하면 이룰 수 있겠지? 끈기있는 태도로 임하고 싶은데 생각보다 쉽지 않다. 자신감을 가질 수 있을까?']

# loss : [3.086986541748047, 0.797802209854126]
# acc :  0.8415841460227966
# 당신에게 들려주고 싶은 이야기는 ['모든 슬픔은 그것을 이야기로 만들거나 그것에 대해 이야기 할 수 있
# 다면 견뎌질 수 있다. 이자크 디녜센']
# project_01_0801_1228_0156-3.9748.hdf5

# loss : [3.578779458999634, 0.6395604610443115]
# acc :  0.7524752616882324
# 당신에게 들려주고 싶은 이야기는 ['인생은 가까이서 보면 비극, 멀리서 보면 희극이다. 찰리 채플린']
# project_01_0801_1238_0018-3.8845.hdf5

# loss : [2.801826238632202, 0.6989011168479919]
# acc :  0.7557755708694458
# 당신에게 들려주고 싶은 이야기는 ['용서가 힘든 것은 나를 넘어서는 일이기 때문이다.']
# project_01_0801_1245_0083-2.7390.hdf5


# + tanh

# epochs=1000, batch_size=100, patience=200
# output_dim=100, activation='tanh'
# x_predict = ['꾸준히 노력하면 이룰 수 있겠지? 끈기있는 태도로 임하고 싶은데 생각보다 쉽지 않다. 자신감을 가질 수 있을까?']

# loss : [2.6504149436950684, 0.7890110015869141]
# acc :  0.8646864891052246
# 당신에게 들려주고 싶은 이야기는 ['당신에게 지금 실패가 있다면 감격해도 좋다. 꾸준히 무엇인가에 
# 도전하고 있다는 증거기 때문이다.']
# project_01_0801_1314_0031-3.1162.hdf5

# loss : [2.397521495819092, 0.8417582511901855]
# acc :  0.8976897597312927
# 당신에게 들려주고 싶은 이야기는 ['사람은 하루에 6만 가지 생각을 하는데 그중 95%가 그 전날 혹은 
# 그 전전날에 했던 생각이다. 그리고 그중 80%가 부정적인 생각이다.']
# project_01_0801_1437_0029-2.6742.hdf5

# loss : [2.280719518661499, 0.8527472615242004]
# acc :  0.9009901285171509
# 당신에게 들려주고 싶은 이야기는 ['승자는 시간을 관리하며 살고 패자는 시간에 끌려 산다. J. 하비스
# ']
# project_01_0801_1442_0033-2.6216.hdf5

# loss : [2.352271795272827, 0.8373626470565796]
# acc :  0.8877887725830078
# 당신에게 들려주고 싶은 이야기는 ['길을 잃는다는 것은 곧, 길을 알게 되는 것이다.']
# project_01_0801_1449_0031-2.8914.hdf5

# loss : [3.043654680252075, 0.8769230842590332]
# acc :  0.9141914248466492
# 당신에게 들려주고 싶은 이야기는 ['오늘이라는 날은 두 번 다시 오지 않는다는 것을 잊지 말라. 단테']
# project_01_0801_1455_0173-2.8678.hdf5

# loss : [2.3941471576690674, 0.8263736367225647]
# acc :  0.8811880946159363
# 당신에게 들려주고 싶은 이야기는 ['가장 바쁜 사람이 가장 많은 시간을 갖는다. 부지런히 노력하는 사
# 람이 결국 많은 대가를 얻는다. 알렉산드리아 피네']
# project_01_0801_1625_0031-2.4889.hdf5

# loss : [2.1247780323028564, 0.8769230842590332]
# acc :  0.9141914248466492
# 당신에게 들려주고 싶은 이야기는 ['전부를 걸 수 있을 만한 사람이 나타났다면 사랑을 시작하라. 긴 
# 인생에서 잘난 당신이 조급해 할 것이 뭐가 있나.']
# project_01_0801_1642_0030-2.5101.hdf5