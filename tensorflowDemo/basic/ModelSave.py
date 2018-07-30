import tensorflow as tf
# v1=tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
# v2=tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
# result=v1+v2
#
# init_op=tf.global_variables_initializer()
# saver=tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init_op)
#     saver.save(sess=sess,save_path="./save/model.ckpt")

'''
保存滑动平均模型
'''
v=tf.Variable(0,dtype=tf.float32,name="v")
for variable in tf.global_variables():
    print(variable.name)
ema=tf.train.ExponentialMovingAverage(0.99)
maintain_average_op=ema.apply(tf.all_variables())
for variable in tf.global_variables():
    print(variable.name)
saver=tf.train.Saver()
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(v,5))
    sess.run(maintain_average_op)
    saver.save(sess,"./save/expotential_model.ckpt")
    print(sess.run([v,ema.average(v)]))