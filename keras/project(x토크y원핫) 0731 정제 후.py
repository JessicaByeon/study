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


# 감정/상태 분류와 글귀를 나누어 각각 처리할 예정
# 감정/상태 분류를 x, 글귀를 y로 나누어 처리

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
print(pad_x1.shape) # (303, 13)

x = pd.DataFrame(pad_x1)
print(x)

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

#########################################################################
# 데이터 추가(220731) 후

# 훈련 10
# x_predict = ['행복하고 싶다, 지금 당장!']
# loss : [5.730127811431885, 0.0]
# acc :  0.013201320543885231
# acc :  0.013201320543885231
# 당신에게 들려주고 싶은 이야기는 ['짊어지는 무게가 아닌, 짊어지는 방법이 중요해요.']

# 훈련 100
# x_predict = ['행복하고 싶다, 지금 당장!']
# loss : [112.69136047363281, 0.0]
# acc :  0.5214521288871765
# 당신에게 들려주고 싶은 이야기는 ['말하지 말고 행동하세요. 말하지 말고 보여주세요. 약속하지 말고 증
# 명하세요.']

# 훈련 500
# x_predict = ['행복하고 싶다, 지금 당장!']
# loss : [5.711215019226074, 0.0]
# acc :  0.0066006602719426155
# 당신에게 들려주고 싶은 이야기는 ['동기부여가 시동을 걸고 습관은 계속 가는 추진력이다. 짐 론']

# 훈련 1000
# x_predict = ['행복하고 싶다, 지금 당장!']
# loss : [5.711954116821289, 0.0]
# acc :  0.0066006602719426155
# 당신에게 들려주고 싶은 이야기는 ['백 권의 책에 쓰인 말보다, 한 가지 성실한 마음이 더 크게 사람을 
# 움직인다. 프랭클린']

#########################################################################
# earlystopping + mcp 적용
# epochs=100, batch_size=20
# loss : [92.80060577392578, 0.0]
# acc :  0.5247524976730347
# 당신에게 들려주고 싶은 이야기는 ['남을 따르는 법을 알지 못하는 사람은 좋은 지도자가 될 수 없다. 아
# 리스토텔레스']

# epochs=100, batch_size=20
# x_predict = ['꾸준히 노력하면 이룰 수 있겠지?']
# loss : [5.712352275848389, 0.0]
# acc :  0.0033003301359713078
# 당신에게 들려주고 싶은 이야기는 ['오늘 당신의 인생을 변화시켜라. 미래에 모험을 걸려고 하지 말고, 
# 지금 행동하라. 지체 없이.']

# epochs=500, batch_size=100
# x_predict = ['꾸준히 노력하면 이룰 수 있겠지? 끈기있는 태도로 임하고 싶은데 생각보다 쉽지 않다. 자신감을 가질 수 있을까?']
# loss : [5.712026596069336, 0.0]
# acc :  0.0033003301359713078
# 당신에게 들려주고 싶은 이야기는 ['위로의 대부분은 과거에서 온다. 과거의 어느 기억, 한 지점에서 우
# 리는 한없이 착해진다.']

# epochs=1000, batch_size=100
# x_predict = ['꾸준히 노력하면 이룰 수 있겠지? 끈기있는 태도로 임하고 싶은데 생각보다 쉽지 않다. 자신감을 가질 수 있을까?']
# loss : [5.711885452270508, 0.0]
# acc :  0.009900989942252636
# 당신에게 들려주고 싶은 이야기는 ['희망을 가져본 적이 없는 자는 절망할 자격도 없다. 버나드 쇼']

# epochs=1000, batch_size=100, patience=100
# x_predict = ['꾸준히 노력하면 이룰 수 있겠지? 끈기있는 태도로 임하고 싶은데 생각보다 쉽지 않다. 자신감을 가질 수 있을까?']
# loss : [5.7124199867248535, 0.0]
# acc :  0.0033003301359713078
# 당신에게 들려주고 싶은 이야기는 ['인생이라는 학교에는 불행이라는 훌륭한 스승이 있다. 그 스승 덕분
# 에 우리는 더욱 단련된다.']

##############################################################
# 비교용 데이터
# epochs=1000, batch_size=100, patience=200
# # x_predict = ['꾸준히 노력하면 이룰 수 있겠지? 끈기있는 태도로 임하고 싶은데 생각보다 쉽지 않다. 자신감을 가질 수 있을까?']

# loss : [2.451578378677368, 0.7890110015869141]
# acc :  0.8646864891052246
# 당신에게 들려주고 싶은 이야기는 ['토끼를 잡으려면 귀를 잡아야 하고, 고양이는 목덜미를 잡아야 한
# 다. 사람을 잡으려면 마음을 잡아야 한다.']

# loss : [5.711859226226807, 0.0]
# acc :  0.0033003301359713078
# 당신에게 들려주고 싶은 이야기는 ['위로의 대부분은 과거에서 온다. 과거의 어느 기억, 한 지점에서 
# 우리는 한없이 착해진다.']
# project_01_0801_1547_0001-5.7123.hdf5

# loss : [5.7119598388671875, 0.0]
# acc :  0.0066006602719426155
# 당신에게 들려주고 싶은 이야기는 ['세계와 나의 어머니를 저울질 한다면 세계 쪽이 훨씬 가벼울 것이
# 다. 랑구랄']
# project_01_0801_1551_0001-5.7126.hdf5

# loss : [5.712050914764404, 0.0]
# acc :  0.0033003301359713078
# 당신에게 들려주고 싶은 이야기는 ['탐욕은 일체를 얻고자 욕심내어서 도리어 모든 것을 잃어버린다. 
# 몽테뉴']
# project_01_0801_1553_0001-5.7119.hdf5


# + tanh
# loss : [5.711846828460693, 0.0]
# acc :  0.013201320543885231
# 당신에게 들려주고 싶은 이야기는 ['인내없는 열정은 광기에 불과하다. 토마스 홉스']
# project_01_0801_1612_0001-5.7120.hdf5

# loss : [5.711946487426758, 0.0]
# acc :  0.013201320543885231
# 당신에게 들려주고 싶은 이야기는 ['세상은 고통으로 가득 차 있지만, 한편 그것을 이겨내는 일로도 가
# 득 차 있다. 헬렌 켈러']
# project_01_0801_1615_0001-5.7121.hdf5
