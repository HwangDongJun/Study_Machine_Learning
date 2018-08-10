#RNN_Hi_Hello_Training에서 단어마다 직접 설정을 하지 말고 자동으로 해보자.
import tensorflow as tf
import numpy as np

sample = " if you want you"
idx2char = list(set(sample)) # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)} # char -> index
#{c: i}는 hash에 해당하며, c는 key / i는 value에 해당한다. i는 숫자, c는 문자

dic_size = len(char2idx) # RNN input size
rnn_hidden_size = len(char2idx) #RNN output size
num_classes = len(char2idx) # final output size
batch_size = 1 # one sample data, one batch
sequence_length = len(sample) - 1 # number of lstm unfolding
#hello에서 hell에 해당하므로, -1을 해준다.

sample_idx = [char2idx[c] for c in sample] # sample의 한 글자씩 가져와서 c에 넣어서 각 문자의 숫자를 추출
x_data = [sample_idx[:-1]] # ex) hello -> hell
y_data = [sample_idx[1:]]  # ex) hello -> ello 에 해당한다.

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, num_classes) # one hot 1 => 0 1 0 0 0 0 0 0 0 0
#여기서 num_classes는 idx2char의 크기와 똑같다.

cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, target=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "Prediction:", ''.join(result_str))