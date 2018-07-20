import tensorflow as tf
import numpy as np
#원하는 데이터가 엄청 많을 경우 파일로 가져오는 예시이다.
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
#delimiter로 ,를 기준으로 xy에 배열로 들어간다. 2차원 배열로 들어간다.
x_data = xy[:, 0:-1] #앞의 :는 행을 의미하며 전체 행을 가져오고 0:-1은 0부터 마지막에서 두번째까지(마지막을 제외)
y_data = xy[:, [-1]] #전체의 행을 가져오며, 마지막만 가지고 오겠다.

#print(x_data.shape, x_data, len(x_data))
#print(y_data.shape, y_data)
#올바른 데이터를 불러왔는지 반드시 확인하는 절차가 필요하다.

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

#이런식으로 다른 점수에 관해 물어볼 수도 있다.
#해당방법은 위의 과정을 지난 후에 실행을 하므로, 학습을 한 후 test한 부분이다.
#print("Your score will be ", sess.run(hypothesis,
#                feed_dict={X: [[100, 70, 101]]}))
#print("Other scroes will be ", sess.run(hypothesis,
#                feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))