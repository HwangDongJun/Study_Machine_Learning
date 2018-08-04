import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sess = tf.InteractiveSession()
image = np.array([[[1],[2],[3]],
                  [[4],[5],[6]],
                  [[7],[8],[9]]], dtype=np.float32) #해당 image는 1이 흰색 9가 검은색으로 9계층의 색깔이 나타난다.
print("image.shape", image.shape)
#print의 경우 (3, 3, 1)이다.
#shape의 경우 (1, 3, 3, 1)은 첫번째 1은 1개의 image사용, 3, 3은 3x3의 크기의 image, 마지막 1은 color에 해당한다.
#plt.imshow(image.reshape(3,3), cmap='Greys')
weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]]) #filter의 경우 다음과 같이 나타난다.
print("weight.shape", weight.shape)
#결과는 (2, 2, 1, 1)인데, 2x2의 filter의 크기와 1은 color로 image의 color와 동일해야하고, 마지막 1은 1개의 filter를 사용한다는 의미이다.
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')
#image와 위에서 만든 weight를 넣고 stride의 경우 가운데 1, 1이 1x1로 stride를 준다.(강의 설명)
conv2d_img = conv2d.eval() #eval()을 통해서 실행시킨다.
print("conv2d_image.shape", conv2d_img.shape)
#결과는 (1, 2, 2, 1)로 결과 모양을 알 수가있다.
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(2,2))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(2,2), cmap='gray')
    #결과로 [[ 12. 16.]
            #[ 24. 28.]] 이다. filter에 걸리는 부분을 모두 더한 값이 나도게 된다. 12의 경우 1+2+4+5이다.
#해당 코드는 결과를 눈으로 보기 위해 실행시켰을 뿐 크게 알 필요는 없다.
#코드의 돌아가는 원리 정도만 알아도 된다.


#conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
#여기서 padding의 값이 바뀐걸 확인이 가능하다. zero padding기능을 tensorflow가 자동으로 구행하게 된다. padding='SAME'이라면.
#이 경우 결과는 [[ 12. 16.  9.]
                #[ 24. 28. 15.]
                #[ 15. 17.  9.]]인 결과가 나온다.