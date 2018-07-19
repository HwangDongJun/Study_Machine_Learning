import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None]) #shape를 옵션으로 주며, 원하는 차원을 만든다.
Y = tf.placeholder(tf.float32, shape=[None]) #2개이상 n개 이상일 수 있다. None는 아무값이나 가능
#직접 값 지정이 아닌 placeholder사용가능

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = X * W + b
#tf.square는 제곱을 합니다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                    feed_dict={X: [1, 2, 3, 4], Y: [2.1, 3.1, 4.1, 5.1]})
    #해당하는 W는 1, b는 1.1에 수렴해야한다.
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)
#해당하는 방법은 동일하게 사용을 해야지 sess.run이 실행이 된다.
#cost_val부분을 지우고 그냥 cost를 사용하면 sess.run되지 않고 정보가 출력

#print(hypothesis, feed_dict={X: [5]})
#print(hypothesis, feed_dict={X:[1.5, 3.5]})
#해당방법으로 hypothesis도 확인이 가능하다.