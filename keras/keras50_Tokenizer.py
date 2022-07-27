from keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text]) # 리스트 형태로 받아들임. 여러개 가능하단 얘기!

print(token.word_index) 
# {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}

# 기본적으로 어구 순서대로 나오지만, 빈도수에 따라 최다 빈도를 가진 단어가 제일 앞으로 위치
# 텍스트를 수치화해서 인식하는 컴퓨터

x = token.texts_to_sequences([text])
print(x)
# [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]

# 수치화 되어있으나 밥을(6)이 나는(3)의 2배의 가치를 갖는 것은 아니기 때문에 평등하게 만들어주기 위해 원핫인코딩을 해줌.

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

x = to_categorical(x)
print(x)
# [[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
print(x.shape) # (1, 11, 9) --> 3차원 : LSTM 

# 자연어처리는 말/글의 문맥을 따라 읽어들여야하므로 시계열 데이터로 처리
# 3차원 LSTM RNN Conv1D


# ===== 사이킷런 원핫인코더 =====


# #3/ sklearn의 OneHotEncoder
import numpy as np
from sklearn.preprocessing import OneHotEncoder
x = np.array(x).reshape(-1,1)
print(x.shape) # (99, 1)

ohe = OneHotEncoder()
ohe.fit(x)

x = ohe.transform(x).toarray()
print(x.shape) # (99, 2)
# # 해당 기능을 통해 y값을 클래스 수에 맞는 열로 늘리는 원핫 인코딩 처리를 한다.
# # 1개의 컬럼으로 [0,1,2] 였던 값을 ([1,0,0],[0,1,0],[0,0,1]과 같은 shape로 만들어줌)