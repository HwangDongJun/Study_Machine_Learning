import tensorflow as tf
#하나의 파일이 아닌 여러개의 파일에서 가져올때, tensorflow에 queue로 가져오는 기능을 사용한다.
filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=false, name='filename_queue')
#여기서는 한개의 파일만 가져왔지만 옆에 추가하면 된다.
reader = tf.TextLineReader()
key, value = reader.read(filename_queue) #key와 value로 가져온다.

record_defaults = [[0.], [0.], [0.], [0.]]#float형태라는 걸 나타낸다.
xy = tf.decode_csv(value, record_defaults=record_defaults)#datatype에 해당한다.

#x와 y의 node이름을 위에서 정해준다.
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)
    #x가 무엇인지, y가 무엇인지 정의해주고 한번 가져올때 size가 10을 정의한다.ㅎ
#queue기능을 사용할 때는 데이터의 크기가 크다는 것을 의미하므로, batch단위로 나눠서 가져오게 된다.
#전체의 데이터를 신경쓰지 않아도 설정만 해두면 tensorflow가 알아서 가져오게 된다.

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

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#이 2줄은 그냥 queue기능 사용할때 사용한다고만 알아둘것.
for step in range(2001):
    #데이터를 펌프질을 해서 가져온다.
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train],
        feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val,
                    "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)
#이 2줄도 일반적으로 알아둘 것.
