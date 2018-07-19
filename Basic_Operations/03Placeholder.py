import tensorflow as tf

#지정된 값이 아닌 그때마다 원하는 값을 넘겨주는 것을 placeholder사용
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

sess = tf.Session()

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
#해당하는 방법으로 feed_dict사용하여 값을 준다.
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))
#2개의 값을 한번에 넘기게 된다.