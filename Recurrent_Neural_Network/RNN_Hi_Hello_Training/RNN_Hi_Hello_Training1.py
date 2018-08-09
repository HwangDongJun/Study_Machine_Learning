#hihello문장을 h를 입력했을 경우 다음 i를 예측하는 것을 만든다.
import tensorflow as tf
import numpy as np

#hihello에서 유니크한 문자를 고른다. h i e l o 이다.
#이것을 index로 정한다. h:0, i:1, e:2, l:3, o:4
#[1,0,0,0,0]=>h:0 / [0,1,0,0,0]=>i:1 / [0,0,1,0,0]=>e:2 / [0,0,0,1,0]=>l:3 / [0,0,0,0,1]=>o:4 으로 One_hot encoding실행
idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0,1,0,2,3,3]] #hihell
x_one_hot = [[[1,0,0,0,0],
              [0,1,0,0,0],
              [0,0,1,0,0],
              [0,0,0,1,0],
              [0,0,0,1,0]]]
y_data = [[1,0,2,3,3,4]] #ihello
#hihello.PNG참고

#hihello.PNG를 보고 입력이 5 / 입력이 1개이니 batch_size가 1 / sequence_length가 6 / hidden_size가 5임을 알 수 있다.
batch_size = 1
sequence_length = 6
hidden_size = 5
X = tf.placeholder(tf.float32, [None, sequence_length, hidden_size])
#batch_size의 경우 None으로 어떤 수라도 가능
Y = tf.placeholder(tf.int32, [None, sequence_length])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,
                        state_is_tuple=True) #state_is_tuple=True는 그냥 넘어가기
initial_satae = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_satae=initial_satae, dtype=tf.float32)
weights = tf.ones([batch_size, sequence_length])
#내 생각에 [batch_size, sequence_length]의 크기만큼이 전부 1값이 된다.

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, target=Y, weigths=weights)
#logits는 결과, target는 Y값을 준다. weight는 tf.ones으로 인해 값을 전부 1로 바꿔서 넣는다.
#간단하게 하기 위해 RNN으로 나온 outputs이 바로 들어갔지만, 원래는 그렇게 하면 안된다.
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
#학습을 진행시킨다.

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction:", result, "true Y:", y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str:", ''.join(result_str))

#loss의 경우 sequence_loss을 이용해서 Cost를 만든다.
#마지막의 Prediction str은 원하는 결과값을 str형태로 보이기 위해 출력한 것이다.