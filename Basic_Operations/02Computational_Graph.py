import tensorflow as tf

#부모노드 하나에 자식노드 둘인 간단한 그래프 구현
node1 = tf.constant(3.0, tf.float32) #3.0뒤에있는것은 datatype이다. 옵션
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2) #두개의 노드를 더하는 부모노드에 해당

#print("node1:", node1, "node2:", node2)
#print("node3:", node3)
#해당하는 출력방법은 그냥 node들의 정보를 출력해주지 값이 나오지 않는다.
sess = tf.Session()
#Session을 이용해서 출력을 해야한다.
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))