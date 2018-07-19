import tensorflow as tf

#학습할 데이터
x_train = [1, 2, 3]
#y_train은 주어진 실제 값
y_train = [1, 2, 3]
#Variable은 tensorflow가 사용하는 변수이다. 기존의 변수선언과는 다르다.
W = tf.Variable(tf.random_normal([1]), name='weight')
#random한값을 주기위해 random_normal을 사용하며, (중요!)shape은 1차원(Rank가 1)인 [1]을 주었다.
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b
#tf.square는 제곱을 합니다.
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
#reduce_mean은 평균을 내주는 함수이다.

#Minimize(중요!)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
#현재로서는 magic으로 알고있고, 간단하게 cost(loss)함수를 최적화(최소화)
#--위에서 그래프를 build했습니다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#global_variables_initializer을 통해서 W,b같은 Variable로 선언한 것을 사용가능

for step in range(2001):
    sess.run(train) #train을 실행시키면서 cost가 최적화가 되면서 진행이 된다.
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
    #2000번을 다 하면 오래걸리기에 제한을 둔다.

#주어진 데이터를 보면 1은 1 / 2는 2 / 3은 3이 된다.
#그럼 여기서 Wx + b에서 생각해보면 W는 1이며, b는 0이 되야한다.
#W와b는 랜던한 값이 나오게 되며, 계속 학습을 해가면서 우리가 원하는 1과 0에 점점 수렴하는 것을 보인다.
#cost는 점점더 작아진다.