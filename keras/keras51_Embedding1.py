from keras.preprocessing.text import Tokenizer
import numpy as np

#1. 데이터
docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '민수가 못 생기긴 했어요',
        '안결 혼해요'
]

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0]) # (14, )

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, 
# '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, 
# '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, 
# '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, 
# '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '민수가': 25, '못': 26, 
# '생기긴': 27, '했어요': 28, '안결': 29, '혼해요': 30}

# 수치화
x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], 
# [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]]
# shape 가 다른 상태 -> 가장 큰 shape에 맞춰 0을 채워줌
# 너무 큰 shape이 있다면 적당히 잘라주고, 작은 shape은 늘려서 길이를 맞춰줌

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) 
# 현재 2가지 문제가 있음. 가장 큰 건 5, 가장 작은 건 1, 
# 0을 채워야하는데 앞에서부터? 뒤에서부터? 어디서부터 채워야할지?
# LSTM 으로 연산을 하기 때문에 통상적으로 앞에 0을 채우고, 뒤의 데이터로 연산되어 수렴되도록 함
# maxlen=5 : 최대 글자를 5글자까지만 하겠다.

print(pad_x)
print(pad_x.shape)
# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  1  5  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]
#  [ 0 25 26 27 28]
#  [ 0  0  0 29 30]]
# (14, 5) -> (14, 5, 1) 로 변경해서 LSTM/Conv1D 사용가능

word_size = len(token.word_index)
print("word_size : ", word_size) # 단어사전의 갯수 : 30

print(np.unique(pad_x, return_counts=True))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]), 0까지 포함해서 unique 값이 31개
#  array([37,  3,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
#       dtype=int64))


# 상관관계가 가까운 쪽에 수치를 줌
# 단어간의 유사도를 파악하여 학습
# 원핫인코딩을 하지 않고 바로 임베딩으로!


#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding # (통상 input layer쪽에서 많이 씀)

model = Sequential()        # 인풋은 (14,5) (원핫을 하지 않은 상태)
                   # 단어사전의 갯수
model.add(Embedding(input_dim=30, output_dim=10, input_length=5)) # -> 통상 3차원 수치 출력
# model.add(Embedding(input_dim=31, output_dim=10)) # -> 통상 3차원 수치 출력
# model.add(Embedding(31, 10)) # 이런 형태로도 사용 가능
# model.add(Embedding(31, 10, 5)) # ValueError: Could not interpret initializer identifier: 5
# model.add(Embedding(31, 10, input_length=5)) # input_length 명시를 해줘야 실행가능

model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid')) # 0 혹은 1로 분류해야하므로 sigmoid

model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 5, 10)             310
# _________________________________________________________________
# lstm (LSTM)                  (None, 32)                5504
# _________________________________________________________________
# dense (Dense)                (None, 1)                 33
# =================================================================
# Total params: 5,847
# Trainable params: 5,847
# Non-trainable params: 0
# _________________________________________________________________

# 임베딩 파라미터 연산 갯수 = 전체 단어사전의 갯수(31) * 출력값(10)
# input length 의 길이를 알면 명시해주고, 몰라서 명시하지 않으면 자동으로 None 으로 (파라미터 연산 개수 연관없음)

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(pad_x, labels, epochs=20, batch_size=16)

#4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]  # [0] 넣으면 loss가 나옴
print('acc : ', acc)


# acc :  0.9285714030265808