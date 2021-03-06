# 함수형 모델구성으로 변경

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
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


#2. 모델구성
# model = Sequential()
# model.add(Dense(100, activation='linear', input_dim=9))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# input1 = Input(shape=(9,)) # 먼저 input layer를 명시해줌
# dense1 = Dense(100, activation='linear')(input1)
# dense2 = Dense(100, activation='relu')(dense1)
# dense3 = Dense(100, activation='relu')(dense2)
# output1 = Dense(1, activation='sigmoid')(dense3)
# model = Model(inputs=input1, outputs=output1)


# #3. 컴파일, 훈련
# from tensorflow.python.keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='loss', patience=200, mode='min', verbose=1, 
#                               restore_best_weights=True) 
# model.compile(loss='binary_crossentropy', optimizer='adam',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=250, batch_size=200, 
#                 validation_split=0.2,
#                 callbacks=[earlyStopping],
#                 verbose=1)

# model.save("./_save/keras23_save_model12_kaggle_titanic.h5")
model = load_model("./_save/keras23_save_model12_kaggle_titanic.h5")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

y_predict [(y_predict <0.5)] = 0  
y_predict [(y_predict >=0.5)] = 1 
# print(y_predict)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 : ', acc) 
# acc 스코어 :  0.7623318385650224

y_summit = model.predict(test_set)

# print(y_summit)
# print(y_summit.shape) # (418, 1)

gender_submission['Survived'] = y_summit
submission = gender_submission.fillna(gender_submission.mean())
submission [(submission <0.5)] = 0  
submission [(submission >=0.5)] = 1  
submission = submission.astype(int)
submission.to_csv(path + 'gender_submission_test01.csv', index=True)

# train_size=0.9, epochs=500, batch_size=200, 
# loss : [0.5329717993736267, 0.8111110925674438]
# acc 스코어 :  0.8111111111111111 / score 0.72488



#=============================================================================
# loss : [0.5881829857826233, 0.800000011920929]
# acc 스코어 :  0.8
#=============================================================================
# MinMaxScaler
# loss : [0.6370605826377869, 0.7777777910232544]
# acc 스코어 :  0.7777777777777778
#=============================================================================
# StandardScaler
# loss : [1.040970802307129, 0.7888888716697693]
# acc 스코어 :  0.7888888888888889
#=============================================================================
# MaxAbsScaler
# loss : [0.6537775993347168, 0.8111110925674438]
# acc 스코어 :  0.8111111111111111
#=============================================================================
# RobustScaler
# loss : [0.9852086305618286, 0.7777777910232544]   
# acc 스코어 :  0.7777777777777778

# 함수형 모델 =================================================================
# MinMaxScaler
# loss : [0.620473325252533, 0.8111110925674438]    
# acc 스코어 :  0.8111111111111111