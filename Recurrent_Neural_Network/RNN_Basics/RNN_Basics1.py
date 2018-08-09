import tensorflow as tf
import numpy as np
import pprint as pp

hidden_size = 2 #우리가 원하는 크기로 지정이 가능하다.
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
#RNN방법중에 쓰이는 방법 중 LSTM이 있다.
#cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
#기존의 RNN방법도 존재하므로, cell부분과 아래의 output부분을 나눠서 구분한다.(원하는 방법이 다르기 때문에)
#hidden_size로 2가 들어가는데, x_data는 1,1,4로 shape가 들어간다면, hidden_size로 인해서 1,1,2가 출력이 된다.
#우리가 정해준 size로 결과가 나오는 것으로 4와 2에 주목하자.

x_data = np.array([[[1,0,0,0]]], dtype=np.float32)
#임의로 입력 데이터 생성
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
#cell과 입력 데이터를 매개변수로 전달. outputs이 나오고, _sataes설명은 나중에

sess = tf.Session()
sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval()) #결과로는 array([[[값, 값]]])의 2짜리가 나온다.