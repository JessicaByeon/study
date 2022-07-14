import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, GRU

a = np.array(range(1,101)) # 1~100
x_predict = np.array(range(96, 106)) # 96~105 / 101~105를 예측
size = 5 # timesteps x는 4개, y는 1개

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size +1): # 10 - 5 + 1 = 6 / range 6 / range 횟수
        subset = dataset[i : (i + size)] # 1:6, 2:7, ... 이런식의 부분집합을 만들자.
        aaa.append(subset) # .append 선택된 요소의 마지막에 새로운 요소나 콘텐츠를 추가
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
# [[  1   2   3   4   5]
#  [  2   3   4   5   6]
#  ...
#   [ 95  96  97  98  99]
#  [ 96  97  98  99 100]]
print(bbb.shape) # (96, 5)


x = bbb[:, :-1]  # 모든 행/열 포함 , 끝에서 -1위치(역순에서 첫번째) 이전의 위치까지
y = bbb[:, -1]  # 모든 행/열 포함 , 끝만! (:이 없으면 역순에서 첫번째만 반환)
print(x, y)
# [[ 1  2  3  4]
#  [ 2  3  4  5]
# ...
#  [95 96 97 98]
#  [96 97 98 99]] [  5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22
#   23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40
#   41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58
#   59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76
#   77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94
#   95  96  97  98  99 100]

print(x.shape, y.shape) # (96, 4) (96,)
# x = x.reshape(96,4,1)

print(x_predict) # [ 96  97  98  99 100 101 102 103 104 105]
print(x_predict.shape) # (10,)

ccc = split_x(x_predict, 4)
print(ccc)
# [[ 96  97  98  99]
#  [ 97  98  99 100]
#  [ 98  99 100 101]
#  [ 99 100 101 102]
#  [100 101 102 103]
#  [101 102 103 104]
#  [102 103 104 105]]

print(ccc.shape) # (7, 4)


#2. 모델구성
model = Sequential()
model.add(Dense(64, input_shape=(4,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
# model.summary

### input_shape=(4,1)로 넣으면 밸류에러 발생 (차원이 3차원이 되므로 충돌)
### ValueError: Input 0 of layer sequential is incompatible with the layer: expected axis -1 of input shape to have value 1 but received input with shape (None, 4)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=100)


#4. 평가, 예측
loss = model.evaluate(x,y)
# y_pred = ccc.reshape(7,4,1)
y_predict = ccc
result = model.predict(y_predict)
print('loss :', loss)
print('result: ', result)

# loss : 0.005028674844652414
# result:  [[100.06392 ]
#  [101.08428 ]
#  [102.10464 ]
#  [103.12499 ]
#  [104.145355]
#  [105.16572 ]
#  [106.18609 ]]

# 데이터가 일정량이 많아지면 LSTM이 더 효과적
# 주가, 환율, 돈, 날씨 등의 시계열 데이터 -> LSTM
# CNN -> RNN, LSTM 가능 모델 호환을 모두 가능하도록, reshape도 할 수 있어야함.
