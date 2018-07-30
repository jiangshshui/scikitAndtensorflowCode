import tensorflow as tf
w1=tf.Variable(tf.random_normal([3,2],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([1,3],stddev=1,seed=1))

x=tf.constant([[0.7],[0.9]])

a=tf.matmul(w1,x)
y=tf.matmul(w2,a)

sess=tf.Session()
# sess.run(w1.initializer)
# sess.run(w2.initializer)
init_variable=tf.global_variables_initializer()
sess.run(init_variable)
print(sess.run(y))
sess.close()
