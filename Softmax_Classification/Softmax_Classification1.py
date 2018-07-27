import tensorflow as tf

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]
#y_data의 경우 ONE-HOT 방법을 사용한다. 3개 중에 한개만 1이고 나머지는 0인 방법이다.

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
#softmax를 사용하여 모든 WX + b의 값을 합계 1인 0과 1사이의 숫자로 바꾼다.

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

#logits = tf.matmul(X, W) + b
#hypothesis = tf.nn.softmax(logits)
#cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels+Y_one_shot)
#cost = tf.reduce_mean(cost_i)
#위의 cost함수 구하는 방법은 17번째줄과 동일한 기능을 하며, 코드만 다르다.
#logits에 tf.matmul(X, W) + b의 값이 들어간다는 것에 주의하자!

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

#a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7 ,9], [1, 3, 4, 3], [1, 1, 0, 1]]})
#print(a, sess.run(tf.arg_max(a, 1)))
#tf.arg_max는 합계 1인 3개의 값중에서 가장 max값을 정해서 알려준다.



#우리는 ONE-HOT방식으로 입력을 했었지만, 그렇지 않을 경우 바꾸어 주어야 한다.
#0~6까지 7가지의 경우가 있다고 가정한다.
#Y = tf.placeholder(tf.int32, [None, 1])
#Y_one_hot = tf.one_hot(Y, nb_classes) => nb_classes는 7이다.
#tf.one_hot은 차원을 안으로 한번더 넣어주게 되는데, [[0][3]] => [[[1000000]], [[0001000]]] 이렇게 된다.
#Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) => [[1000000], [0001000]]으로 원래의 차원으로 돌아온다.