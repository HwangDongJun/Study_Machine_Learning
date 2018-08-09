import tensorflow as tf
import numpy as np
import pprint as pp

hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
x_data = np.array([[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,1,0], [0,0,0,1]]], dtype=np.float32)
#One_hot으로 h e l l o 를 h는 1,0,0,0 / e는 0,1,0,0 ~ 으로 만든다.
#모양은 1,5,4로 4의 입력값과 5개의 값으로 나누어진다. 이 길이를 sequence_length라고한다.
#sequence_length로 인해 RNN의 모양에서 옆으로 5개의 값이 생성이 되는 것이고, tf도 자동을 출력값을 5로 해준다.
#결과는 1,5,2로 5는 sequence_length로, 2는 hidden_size로 인해 생성.
print(x_data.shape)
pp.pprint(x_data)

outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())

