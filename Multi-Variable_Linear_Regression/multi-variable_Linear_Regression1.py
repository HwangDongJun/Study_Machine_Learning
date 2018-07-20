import tensorflow as tf
#matrix를 이용하여 실질적으로 유용한 실습을 진행한다.
#해당 예시의 경우 n개의 데이터가 왔을 경우 매우 안좋은 형태를 가지기에 아직 완벽하진 않다.
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    #아래의 코드에서 결국에는 train을 돌리게 되면 자동으로 cost를 최적화를 시키는 과정을 이어간다.
    #cost와 hypothesis는 출력결과를 확인하고 싶어서 train과 같이 돌리는 것뿐이다.
    #실제로 print로 보는것은 cost와 hypothesis뿐이다.
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
    #결과값이 최종 y_data와 유사한것을 볼 수 있다.