<<<<<<< HEAD
import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0]) # 5

import matplotlib.pyplot as plt
plt.imshow(x_train[5], 'gray')
=======
import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0]) # 5

import matplotlib.pyplot as plt
plt.imshow(x_train[5], 'gray')
>>>>>>> 0032b7bc9af5be1bd054bebd127a94da7509d68b
plt.show()