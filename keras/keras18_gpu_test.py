# cpu, gpu 제대로 설치된 부분에 대한 확인 테스트

import numpy as np
import tensorflow as tf

print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) :
    print('쥐피유 돈다')
else:
    print('쥐피유 안도라')