import numpy as np
import tensorflow as tf

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
#XOR문제에 대해 numpy를 사용하여 배열을 할당한다. ex) 0, 0일때는 0이다.
#XOR은 왜 logistic regression일까? 0과 1로 결과가 나오므로, 굳이 softmax를 사용하여 복잡한 연산을 추가하지 않고,
#두가지 분류의 값을 사용하는 logistic regression을 사용한다.

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([2, 1]), name="weight") #입력이 2개, 출력이 1개이므로 [2, 1]이다.
b = tf.Variable(tf.random_normal([1]), name="bias") #bias는 항상 출력의 개수와 같다.

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
#cast는 0.5보다 크면 True => 1 이며, 0.5보다 작으면 False => 0 이다.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("Hypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)

#해당 코드는 만족스럽게 돌아가지 않는다.
#XOR의 경우 1개의 W, b인 Net으로는 원하는 결과를 얻지 못합니다.
#결과적으로 여러개의 Net을 이용하여 입력으로 넣어주어야 하는데, 그것이 Neural Net이다.