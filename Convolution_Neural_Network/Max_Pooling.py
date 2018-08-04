import tensorflow as tf
import numpy as np

image = np.array([[[[4],[3],
                    [[2],[1]]]]], dtype=np.float32)
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                      strides=[1, 1, 1, 1], padding='SAME')
#ksize는 filter의 크기 2x2이며, stride의 경우 1x1로 1이다.
print(pool.shape)
print(pool.eval())
#결과는 (1, 2, 2, 1)이며, 각 값이 4. 3.
                                # 2. 1. 이다. []는 생략
#zero padding과 max_pooling과정을 모두 실행한 결과이다.

#★주의할점! 만약 28x28의 image라면 여기서 stride가 1, 2, 2, 1이면 2이다.
#28의 image가 2씩 진행이 되기에 zero padding로 그대로 유지가 되면 14x14이다.★