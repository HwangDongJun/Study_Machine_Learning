import numpy as np
import tensorflow as tf

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#W = tf.Variable(tf.random_normal([2, 1]), name="weight")
#b = tf.Variable(tf.random_normal([1]), name="bias")
#For_XOR1과의 차이점-------------------------------------
W1 = tf.Variable(tf.random_normal([2, 2]), name="weight1")
b1 = tf.Variable(tf.random_normal([2]), name="bias1")
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([2, 1]), name="weight2")
b2 = tf.Variable(tf.random_normal([1]), name="bias2")
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)
#layer1이 입력으로 들어가며, 2개의 Net을 사용하는 Neural Net을 이용한 것이다.
#Neural_Net_XOR.PNG참고

#★Wide하게 넓힐 수 있는데, W1-> 2, 2 => 2, 10 / 2 => 10, W2-> 2, 1 => 10, 1 으로 만든다.
#★Deep하게 갈 수도 있다.(Deep Learning)
#★W1 => 2, 10 / 10, W2 => 10, 10 / 10, W3 => 10, 10 / 10, W4 => 10, 1 / 1 으로 만들 수 있다.
#★항상 W1은 W2의 입력, W2는 W3의 입력, W3는 W4의 입력에 주의하자!
#깊게 학습할수록 더욱 정확한 값이 나온다. cost도 더 줄어든다.

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("Hypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)