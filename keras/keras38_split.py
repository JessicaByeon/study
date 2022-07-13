import numpy as np

a = np.array(range(1,11)) # 1~10 / [ 1  2  3  4  5  6  7  8  9 10]
size = 5 # timesteps

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size +1): # 10 - 5 + 1 = 6 / range 6 / range 횟수
        subset = dataset[i : (i + size)] # 1:6, 2:7, ... 이런식의 부분집합을 만들자.
        aaa.append(subset) # .append 선택된 요소의 마지막에 새로운 요소나 콘텐츠를 추가
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
print(bbb.shape) # (6, 5)

x = bbb[:, :-1]  # 모든 행/열 포함 , 끝에서 -1위치(역순에서 첫번째) 이전의 위치까지
y = bbb[:, -1]  # 모든 행/열 포함 , 끝만! (:이 없으면 역순에서 첫번째만 반환)
print(x, y)
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]] [ 5  6  7  8  9 10]
print(x.shape, y.shape) # (6, 4) (6,)
