import tensorflow as tf

weight = tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1.]]],
                      [[[1.,10.,-1.]],[[1.,10.,-1.]]]])
#다음과 같은 방법으로 weight만 고쳐서 여러개의 filter를 사용하겠다고 선언할 수도있다.
print(weight.shpae) #(2, 2, 1, 3)이며, 3개의 filter를 사용하겠다고 말한다. 2x2의 크기의 3가지(1. / 10. / -1.)의 값이기 때문에 3개의 filter이다.

#바꾸는 부분에 관해서만 코드를 적었다.