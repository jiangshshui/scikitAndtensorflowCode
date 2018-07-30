import tensorflow as tf
# v1=tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
# v2=tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
# result=v1+v2
# saver=tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess=sess,save_path="./save/model.ckpt")
#     print(sess.run(result))

'''
不需要重复定义模型的参数
'''
# saver=tf.train.import_meta_graph("./save/model.ckpt.meta")
# with tf.Session() as sess:
#     saver.restore(sess,save_path="./save/model.ckpt")
#     print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))

'''
只加载部分指定的参数
'''
# v1=tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
# v2=tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
# result=v1+v2#由于v2没有加载进来,所以使用v2会报错
# saver=tf.train.Saver([v1])#只加载指定的变量
# with tf.Session() as sess:
#     saver.restore(sess=sess,save_path="./save/model.ckpt")
#     print(sess.run(result))

'''
加载变量时,支持重命名操作
'''
# v1=tf.Variable(tf.constant(1.5,shape=[1]),name="other-v1")
# v2=tf.Variable(tf.constant(3.5,shape=[1]),name="other-v2")
# res2=v1+v2
# saver=tf.train.Saver({"v1":v1,"v2":v2})
# with tf.Session() as sess:
#     saver.restore(sess=sess,save_path="./save/model.ckpt")
#     print(sess.run(res2))

'''
加载滑动平均模型的值
'''
# v=tf.Variable(0,dtype=tf.float32,name="v")
# saver=tf.train.Saver({"v/ExponentialMovingAverage":v})
# with tf.Session() as sess:
#     saver.restore(sess,"./save/expotential_model.ckpt")
#     print(sess.run(v))

v=tf.Variable(0,dtype=tf.float32,name="v")
ema2=tf.train.ExponentialMovingAverage(0.1)
maintain_average_op=ema2.apply([v])
print(ema2.variables_to_restore())

saver=tf.train.Saver(ema2.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess,"./save/expotential_model.ckpt")
    print(sess.run(v))
