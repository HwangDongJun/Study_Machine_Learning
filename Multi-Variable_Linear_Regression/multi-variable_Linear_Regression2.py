import tensorflow as tf
#Matrix사용하는 방법
#multi-variable_Linear_Regression1과 비교하면 코드가 확실히 적다.
x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

X = tf.placeholder(tf.float32, shape=[None, 3]) #None란 n을 의미한다. 미지의 수
#x_ata에서 5개뿐이지만 n개로 생각을하고 None를 주고, 한 칸당 3개이므로 3이다.
Y = tf.placeholder(tf.float32, shape=[None, 1])
#XW = Y의 형태에서 matrix는 X가 [n, 3], Y가 [n, 1]이다.
#그러면 W는 [3, 1]이라는걸 추측할 수 있다. X의 3과 Y의 1로 인해.
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b #matmul은 matrix형태로 계산을 해준다.
#H(x) = Wx + b인 형태가 matrix를 사용하는 경우 암묵적으로 XW로 위치의 변경과 대문자의 사용을 한다.(기억!)

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