import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#W를 사용했기 때문에

W_val = list()
cost_val = list()
for i in range(-30, 50):
    feed_W = i * 0.1 #-3~5간격으로 움직이겠다.
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    #0.1간격으로 움직이겠다. feed_dict={W: feed_W}
    W_val.append(curr_W)
    cost_val.append(curr_cost)

#Show the cost function
plt.plot(W_val, cost_val)
plt.show()