import tensorflow as tf
#sigmoid를 사용하여 진행을 합니다. => g(z) = 1 / 1 + e^-z
#여기서 H(X)의 경우 X = WX이므로, H(X) = 1 / 1 + e^-WX 입니다.
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]] #linear regression이 아니기 때문에 0과 1인 binary classification이 주어진다.
#0은 fail, 1은 pass가 될 수도있다.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
#H(X) = 1 / 1 + e^-WX에 해당한다.

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
#해당하는 cost는 sigmod를 사용한 cost함수에 해당합니다.

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#최적화(훈련시킨다.) => gradient descent한다.  // 직접 미분할 필요가 없다.

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#예측한 값을 가지고 0인지 1인지 확인을 해야한다. ex) 0.8이다. 그러면 1에 가깝기 때문에 pass이다.
#중간값이 0.5에 해당한다. hypothesis > 0.5이냐가 T/F로 나오는데 dtype이 tf.float32으로 tf.cast(캐스트)되기 때문에 1.0 아니면 0.0으로 나오게 된다.
#accuracy는 tf.equal(predicted, Y) 우리가 예측한 값과 Y의 값이 똑같은지 T/F로 나오게 되며,
#dtype이 tf.float32으로 tf.cast(캐스트)되기 때문에 1.0아니면 0.0이 된다. tf.reduce_mean으로 확률을 구하게 된다.

#학습하는 model----
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
#h가 예측된 값을 출력하며, c는 0.5를 기준점으로 0아니면 1로 반환을하고,
#h와 c의 값이 얼마나 맞았는지 a가 출력한다. 다 맞았다면 1.0(type이 float이기때문에)을 출력