import tensorflow as tf

#training data와 test data는 실질적인 구별이 반드시 필요하다.
x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5], [1, 7, 5], [1, 2 ,5], [1, 6, 6], [1, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]
#ONE-HOT방법 사용
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
#해당 data는 test data로 training할때는 절대 사용하지 않는다.

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.random_normal([3, 3]), name="weight")
b = tf.Variable(tf.random_normal([3]), name="bias")

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.arg_max(hypothesis, 1) #예측한 값이 있는지. hypothesis를 예측한다.
is_correct = tf.equal(prediction, tf.arg_max(Y, 1)) #예측한 값이 맞는지 틀린지를 측정
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) #측정한 것을 평균을 낸다.

#placeholder의 좋은점은 training이냐 test냐에 따라 원하는 data를 던져주기만 하면 된다.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer],
                        feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)

    # predict
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
    #Calculate the accuracy
    print("Accuracy:", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
    #accuracy의 경우 가장 큰 값이 1.0이다. test data는 한번도 보지 못한 경우 이기 때문에 1.0이면 매우 훌륭한 결과이다.