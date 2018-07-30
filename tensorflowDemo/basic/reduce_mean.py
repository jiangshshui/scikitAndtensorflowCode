import tensorflow as tf
constant_float_ext = tf.constant([[[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]],
                                  [[4.,4.,4.],[5.,5.,5.],[6.,6.,6.]]])
print(constant_float_ext.shape)     #shape=2,3,3
sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tf.reduce_mean(constant_float_ext)))
#0是最外面的维度  axis=0  最外面的维度变化为0,1,其余的对应的保持不变
print(sess.run(tf.reduce_mean(constant_float_ext,axis=0)))
#axis=1 维度的变化取值为0,1,2
print(sess.run(tf.reduce_mean(constant_float_ext,axis=1)))
print(sess.run(tf.reduce_mean(constant_float_ext,axis=2)))
print(sess.run(tf.rank(constant_float_ext)))
sess.close()