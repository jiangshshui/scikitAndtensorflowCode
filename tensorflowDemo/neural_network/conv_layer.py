import numpy as np
import tensorflow as tf

class ConvLayer:
    def __init__(self,input_shape,filter_size,filter_numbers,
                 stride=1,activation='relu',batch_normal=False,
                 weight_decay=None,name='cnov'):
        self.input_shape=input_shape
        self.filter_size=filter_size
        self.activation=activation
        self.stride=stride
        self.batch_normal=batch_normal
        self.weight_decay=weight_decay

        self.weight=tf.Variable(
            initial_value=tf.truncated_normal(shape=[filter_size,filter_size,input_shape[3],filter_numbers],
                                              dtype=tf.float32,mean=0.0,
                                              stddev=np.sqrt(2.0/input_shape[0]*input_shape[1]*input_shape[2])),
            name="W_%s"%(name)
        )

        if self.weight_decay:
            weight_decay=tf.multiply(tf.nn.l2_loss(self.weight),self.weight_decay)
            tf.add_to_collection("losses",weight_decay)

        self.bias=tf.Variable(
            initial_value=tf.constant(0.0,shape=[filter_numbers]),
            name='b_%s'%(name)
        )

        if batch_normal:
            self.epsion=1e-5
            self.gamma=tf.Variable(
                initial_value=tf.constant(1.0,shape=[self.filter_size]),
                name='gamma_%s'%(name)
            )

    def get_layer(self,input):
        self.conv=tf.nn.conv2d(
            input=input,
            filter=self.weight,
            strides=[1,self.stride,self.stride,1],
            padding='SAME'
        )

        self.output_shape=self.conv.get_shape().as_list()

        if self.batch_normal:
            mean,variance=tf.nn.moments(self.conv,axes=[0,1,2],keep_dims=False)
            self.hidden=tf.nn.batch_normalization(self.conv,mean,variance,self.bias,self.gamma,self.epsion)
        else:
            self.hidden=self.conv+self.bias

        if self.activation=='relu':
            self.layer=tf.nn.relu(self.hidden)
        if self.activation=='tanh':
            self.layer=tf.nn.tanh(self.hidden)
        elif self.activation=='None':
            self.layer=self.hidden

        return self.layer

