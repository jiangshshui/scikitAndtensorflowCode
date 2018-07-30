
import tensorflow as tf
import numpy as np

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)



'''
未使用tensorflow的rnn
'''
# reset_graph()
# n_inputs=3
# n_neurons=5
#
# x0_inputs=tf.placeholder(tf.float32,shape=[None,n_inputs])
# x1_inputs=tf.placeholder(tf.float32,shape=[None,n_inputs])
#
# W_x=tf.Variable(tf.random_normal(shape=[n_inputs,n_neurons],dtype=tf.float32))
# W_y=tf.Variable(tf.random_normal(shape=[n_neurons,n_neurons],dtype=tf.float32))
#
# b=tf.Variable(tf.constant(0.,shape=[1,n_neurons],dtype=tf.float32))
#
# x0_batch=np.array([[0,1,2],
#                    [3,4,5],
#                    [6,7,8],
#                    [9,0,1]])
#
# x1_batch=np.array([[9,8,7],
#                    [0,0,0],
#                    [6,5,4],
#                    [3,2,1]])
#
# y0=tf.tanh(tf.matmul(x0_inputs,W_x)+b)
# y1=tf.tanh(tf.matmul(x1_inputs,W_x)+tf.matmul(y0,W_y)+b)
#
# init=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     y0_val,y1_val=sess.run([y0,y1],feed_dict={
#         x0_inputs:x0_batch,
#         x1_inputs:x1_batch
#     })
#     print(y0_val)
#     print(y1_val)


'''
使用tensorflow 的static rnn 的api
'''
# reset_graph()
# n_inputs=3
# n_neurons=5
#
# x0=tf.placeholder(tf.float32,shape=[None,n_inputs])
# x1=tf.placeholder(tf.float32,shape=[None,n_inputs])
#
# x0_batch=np.array([[0,1,2],
#                    [3,4,5],
#         z           [6,7,8],
#                    [9,0,1]])
#
# x1_batch=np.array([[9,8,7],
#                    [0,0,0],
#                    [6,5,4],
#                    [3,2,1]])
#
#
# basic_rnn=tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# output_seqs,states=tf.contrib.rnn.static_rnn(basic_rnn,[x0,x1],dtype=tf.float32)
#
# Y0,Y1=output_seqs
#
# init=tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     y0_val,y1_val,states_val=sess.run([Y0,Y1,states],feed_dict={
#         x0:x0_batch,
#         x1:x1_batch
#     })
#
#     print(y0_val)
#     print(y1_val)
#     print(states_val)
#     print(y1_val==states_val)



'''
input 的shape=[None,n_steps,n_inputs] 的情况
'''
# reset_graph()
# n_steps=2
# n_inputs=3
# n_neurons=5
# x=tf.placeholder(tf.float32,shape=[None,n_steps,n_inputs])
# x_transpose=tf.transpose(x,perm=[1,0,2])
# x_seqs=tf.unstack(x_transpose)
# basic_cell=tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# output_seqs,states=tf.contrib.rnn.static_rnn(basic_cell,x_seqs,dtype=tf.float32)
# output_seqs_stack=tf.stack(output_seqs)
# outputs=tf.transpose(output_seqs_stack,perm=[1,0,2])
#
# x_batch=np.array([
#         # t0         t1
#         [[0, 1, 2], [9, 8, 7]], # instance 1
#         [[3, 4, 5], [0, 0, 0]], # instance 2
#         [[6, 7, 8], [6, 5, 4]], # instance 3
#         [[9, 0, 1], [3, 2, 1]], # instance 4
# ])
#
# init=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     x_transpose_val,x_seqs_val,output_seqs_val,outputs_val=sess.run([x_transpose,x_seqs,output_seqs,outputs],feed_dict={
#         x:x_batch
#     })
#     print(x_transpose_val)
#     print(x_seqs_val)
#     print(output_seqs_val)# 是各个时间序列的结果  t0 t1 t3 ......  t* 中包含的是每个batch_size * n_neurons
#     print(outputs_val)


'''
使用dynamic_rnn api
'''
reset_graph()
n_steps=2
n_inputs=3
n_neurons=5
x=tf.placeholder(tf.float32,shape=[None,n_steps,n_inputs])

basic_cell=tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs,states=tf.nn.dynamic_rnn(basic_cell,x,dtype=tf.float32)


x_batch=np.array([
        # t0         t1
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
])

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    output_seqs_val=sess.run([output_seqs,],feed_dict={
        x:x_batch
    })
    print(output_seqs_val)# 是各个时间序列的结果  t0 t1 t3 ......  t* 中包含的是每个batch_size * n_neurons








