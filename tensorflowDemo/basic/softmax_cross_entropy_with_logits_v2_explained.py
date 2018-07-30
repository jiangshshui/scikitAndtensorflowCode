import tensorflow as tf
import numpy as np
print(tf.__version__)
y_true=tf.convert_to_tensor(np.array([[0.0,1.0,0.0],
                                     [0.0,0.0,1.0]]))
y_hat=tf.convert_to_tensor(np.array([[0.5,1.5,0.1],
                                     [2.2,1.3,1.7]]))

y_hat_softmax=tf.nn.softmax(y_hat)

y_cross=y_true*tf.log(y_hat_softmax)

result=-tf.reduce_sum(y_cross,1)

result_tf=tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat,labels=y_true)

with tf.Session() as sess:
    sess.run(result)
    sess.run(result_tf)
    print('y_hat_softmax:\n{0}\n'.format(y_hat_softmax.eval()))
    print('y_true: \n{0}\n'.format(y_true.eval()))
    print('y_cross: \n{0}\n'.format(y_cross.eval()))
    print('result: \n{0}\n'.format(result.eval()))
    print('result_tf: \n{0}'.format(result_tf.eval()))


p = tf.placeholder(tf.float32, shape=[None, 5])
logit_q = tf.placeholder(tf.float32, shape=[None, 5])
q = tf.nn.sigmoid(logit_q)

feed_dict = {
  p: [[0, 0, 0, 1, 0],
      [1, 0, 0, 0, 0]],
  logit_q: [[0.2, 0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.2, 0.1, 0.1]]
}

prob1 = -p * tf.log(q)
prob2 = p * -tf.log(q) + (1 - p) * -tf.log(1 - q)
prob3 = p * -tf.log(tf.sigmoid(logit_q)) + (1-p) * -tf.log(1-tf.sigmoid(logit_q))
prob4 = tf.nn.sigmoid_cross_entropy_with_logits(labels=p, logits=logit_q)
with tf.Session() as sess:
    print(prob1.eval(feed_dict))
    print(prob2.eval(feed_dict))
    print(prob3.eval(feed_dict))
    print(prob4.eval(feed_dict))

'''
https://stackoverflow.com/questions/49377483/about-tf-nn-softmax-cross-entropy-with-logits-v2
https://stackoverflow.com/questions/46291253/tensorflow-sigmoid-and-cross-entropy-vs-sigmoid-cross-entropy-with-logits
https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow/47034889#47034889


https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks/9311#9311

https://stackoverflow.com/questions/37312421/tensorflow-whats-the-difference-between-sparse-softmax-cross-entropy-with-logi
'''

