import tensorflow as tf
import numpy as np
class DeviceCellWrapper(tf.contrib.rnn.BasicRNNCell):
    def __init__(self,device,cell):
        self._device=device
        self._cell=cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self,inputs,state,scope=None):
        with tf.device(self._device):
            return self._cell(inputs,state,scope)


n_inputs=5
n_steps=20
n_neurons=100

X = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs])

devices = ["/cpu:0", "/cpu:0", "/cpu:0"] # replace with ["/gpu:0", "/gpu:1", "/gpu:2"] if you have 3 GPUs
cells = [DeviceCellWrapper(dev,tf.contrib.rnn.BasicRNNCell(num_units=n_neurons))
         for dev in devices]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(sess.run(outputs, feed_dict={X: np.random.rand(2, n_steps, n_inputs)}))



'''
Alternatively, since TensorFlow 1.1, you can use the tf.contrib.rnn.DeviceWrapper class 
(alias tf.nn.rnn_cell.DeviceWrapper since TF 1.2).
'''
