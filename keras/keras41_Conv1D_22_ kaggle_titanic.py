from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm_notebook

#1. 데이터

path = './_data/kaggle_titanic/' # 경로 = .현재폴더 /하단
train_set = pd.read_csv(path + 'train.csv', # train.csv 의 데이터가 train set에 들어가게 됨
                        index_col=0) # 0번째 컬럼은 인덱스로 지정하는 명령
test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)

# print(train_set)
# print(train_set.shape) # (891, 11) 원래 열이 12개지만, id를 인덱스로 제외하여 11개

# print(train_set.columns)
# print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
# print(train_set.describe())

print(test_set)
print(test_set.shape) # (418, 10) # 예측 과정에서 쓰일 예정


# 결측치 처리
print(train_set.isnull().sum()) # 각 컬럼당 null의 갯수 확인가능 -- age 177, cabin 687, embarked 2
# Survived      0
# Pclass        0
# Name          0
# Sex           0
# Age         177
# SibSp         0
# Parch         0
# Ticket        0
# Fare          0
# Cabin       687
# Embarked      2
# dtype: int64
train_set = train_set.fillna(train_set.median())
print(test_set.isnull().sum())
# Pclass        0
# Name          0
# Sex           0
# Age          86
# SibSp         0
# Parch         0
# Ticket        0
# Fare          1
# Cabin       327
# Embarked      0
# dtype: int64

drop_cols = ['Cabin']
train_set.drop(drop_cols, axis = 1, inplace =True)
test_set = test_set.fillna(test_set.mean())
train_set['Embarked'].fillna('S')
train_set = train_set.fillna(train_set.mean())

print(train_set) 
print(train_set.isnull().sum())

test_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['Name','Sex','Ticket','Embarked']
for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
    
x = train_set.drop(['Survived'],axis=1) #axis는 컬럼 
print(x) #(891, 9)
y = train_set['Survived']
print(y.shape) #(891,)

gender_submission = pd.read_csv(path + 'gender_submission.csv', #예측에서 쓰일 예정
                       index_col=0)

# print(pd.Series.value_counts()) 

x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                    train_size=0.9, shuffle=True, random_state=68)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # 수치로 변환해주는 걸 x_train에 집어넣자.
x_test = scaler.transform(x_test) 
# print(np.min(x_train))
# print(np.max(x_train))
# print(np.min(x_test))
# print(np.max(x_test))

print(x_train.shape) # (801, 9)
print(x_test.shape) # (90, 9)

x_train = x_train.reshape(801, 9, 1)
x_test = x_test.reshape(90, 9, 1)

print(np.unique(y_train, return_counts=True))


#2. 모델구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(9, 1)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# date = datetime.datetime.now()
# # print(date)
# date = date.strftime("%m%d_%H%M")
# # print(date) 

# # 파일명을 계속적으로 수정하지 않고 고정시켜주기 위해
# filepath = './_ModelCheckPoint/k24/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # d4 네자리까지, .4f 소수넷째자리까지

earlyStopping =EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1, 
                             restore_best_weights=True) 

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
#                       save_best_only=True,
#                       filepath= "".join([filepath, 'k24_', date, '_', filename] # .join안에 있는 모든 문자열을 합치겠다.
#                       ))

hist = model.fit(x_train, y_train, epochs=250, batch_size=200, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

y_predict [(y_predict <0.5)] = 0  
y_predict [(y_predict >=0.5)] = 1 
# print(y_predict)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc)

# loss : [0.46116819977760315, 0.7444444298744202]
# acc 스코어 :  0.7444444444444445

# dropout 사용 결과값
# loss : [0.456948846578598, 0.7888888716697693]
# acc 스코어 :  0.7888888888888889

# cnn ==============================================================================
# MinMaxScaler
# loss : [0.5908742547035217, 0.8666666746139526]
# acc 스코어 :  0.8666666666666667

# Conv1D
