<<<<<<< HEAD
import numpy as np
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

print(x_train[0])
print(y_train[0]) # [6]

import matplotlib.pyplot as plt
plt.imshow(x_train[5], 'gray')
plt.show()
=======
import numpy as np
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

print(x_train[0])
print(y_train[0]) # [6]

import matplotlib.pyplot as plt
plt.imshow(x_train[5], 'gray')
plt.show()
>>>>>>> 0032b7bc9af5be1bd054bebd127a94da7509d68b
