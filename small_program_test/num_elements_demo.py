import tensorflow as tf
import numpy as np
# a=tf.convert_to_tensor(np.arange(12).reshape(3,4))
# print(a.shape)
# print(a)
# print(a.num_elements())

# a=tf.Variable(tf.truncated_normal(shape=[2,3,4],stddev=0.05))
# print(a)
# #print(a.num_elements)
#
# arr=np.arange(6*4).reshape(2,3,4)
# print(arr)
# arr=arr.reshape(-1,4)
# print(arr)


with tf.device("/cpu:0"):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
# 新建session with log_device_placement并设置为True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# 运行这个 op.
print(sess.run(c))