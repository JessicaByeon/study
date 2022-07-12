from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y) #y는 데이터 전처리를 할 필요가 없음.
print(x.shape, y.shape) #(442, 10) (442,) Number of Instances: 442, Number of Attributes 10

print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.9, shuffle=True, random_state=72)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) # 수치로 변환해주는 걸 x_train에 집어넣자.
x_test = scaler.transform(x_test) 
# print(np.min(x_train))
# print(np.max(x_train))
# print(np.min(x_test)) 
# print(np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dropout(0.3)) # 30% 만큼 제외
model.add(Dense(7))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(10))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(15))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(12))
model.add(Dropout(0.2)) # 20% 만큼 제외
model.add(Dense(1))


# #3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
# print(date)
date = date.strftime("%m%d_%H%M")
# print(date) 

 # 파일명을 계속적으로 수정하지 않고 고정시켜주기 위해
filepath = './_ModelCheckPoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # d4 네자리까지, .4f 소수넷째자리까지

earlyStopping =EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, 
                             restore_best_weights=True) 

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
                      save_best_only=True,
                      filepath= "".join([filepath, 'k24_', date, '_', filename] # .join안에 있는 모든 문자열을 합치겠다.
                      ))

hist = model.fit(x_train, y_train, epochs=100, batch_size=20, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# loss :  2002.0338134765625
# r2 스코어 :  0.6536007576101125

# dropout 사용 결과값
# loss :  2793.310302734375
# r2 스코어 :  0.5166912102905015