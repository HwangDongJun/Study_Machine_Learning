import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")
sess = tf.Session()
print(sess.run(hello))
#가장 기본적인 출력을 하는 방법에 해당합니다.