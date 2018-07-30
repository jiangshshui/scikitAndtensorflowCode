import tensorflow as tf
import numpy as np

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


n_inputs=3
n_neurons=5
n_steps=2

x=tf.placeholder(tf.float32,shape=[None,n_steps,n_inputs])
seq_length=tf.placeholder(tf.int32,[None])
basic_rnn=tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs,state=tf.nn.dynamic_rnn(basic_rnn,x,dtype=tf.float32,sequence_length=seq_length)


x_batch=np.array([
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2 (padded with zero vectors)
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]]])

seq_length_batch=np.array([2,1,2,2])
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    outputs_val,state_val=sess.run([outputs,state],feed_dict={
        x:x_batch,
        seq_length:seq_length_batch
    })
    print(outputs_val)
    print(state_val)


