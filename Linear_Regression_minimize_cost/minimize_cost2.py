import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_sum(tf.square(hypothesis - Y))

#Gradient descent를 사용한다. 수동으로 학습minimize에 해당
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X) #cost함수를 미분한 식을 나타낸다.
descent = W - learning_rate * gradient
#W에서 빼는 이유는 미분을 함으로써 그 점에서의 기울기가 나오는데, 기울기가 양수(오른쪽 위로)이면 W를 좀더 -쪽으로 보내야한다.
#반면에 기울기가 음수(오른쪽 아래)이면 W를 좀 더 오른쪽으로 보내야 한다.
#learning_rate * gradient가 기울기에 해당한다. minimize_cost그래프(minimize_cost1.py)를 기준으로 생각하면된다.
update = W.assign(descent) #assign으로 미분한 값은 넣어야한다. 공식처럼 생각

############################################################################
#Minimize자동에 해당한다.(어떤 데이터가 들어올지 모르기 때문에 이번 예제는 간단한 데이터)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#train = optimizer.minimize(cost)
#############################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer()) #변수 초기화
for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data}) #x_data, y_data를 던져주면서 update를 실행
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))