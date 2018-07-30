import numpy as np
import tensorflow as tf

class DenseLayer:
    def __init__(self,input_shape,hidden_dim,activation='relu',dropout=False,
                 keep_probe=None,batch_normal=False,weight_decay=None,name='dense'):
        self.input_shape=input_shape
        self.hidden_dim=hidden_dim
        self.activation=activation
        self.weight_dacay=weight_decay
        self.dropout=dropout
        self.batch_normal=batch_normal


        self.weight=tf.Variable(
            initial_value=tf.truncated_normal(shape=[self.input_shape[1],self.hidden_dim],
                                              mean=0.0,
                                              stddev=np.sqrt(2.0/self.input_shape[1])),
            name='W_%s'%(name)
        )

        if weight_decay:
            weight_decay=tf.multiply(tf.nn.l2_loss(self.weight),self.weight_decay)
            tf.add_to_collection("losses",weight_decay)

        self.bias=tf.Variable(initial_value=tf.constant(0.05,shape=[self.hidden_dim]),
                              name='b_%s'%(name))

        if self.batch_normal:
            self.epsion=1e-5
            self.gamma=tf.Variable(initial_value=tf.constant(1.0,shape=[self.hidden_dim]),
                                   name="gamma_%s"%(name))
        if self.dropout:
            self.keep_prob=keep_probe


    def get_layer(self,input):
        self.output_shape=self.weight.get_shape().as_list()

        intermediate=tf.matmul(input,self.weight)

        if self.batch_normal:
            mean,variance=tf.nn.moments(intermediate,axes=[0])
            self.hidden=tf.nn.batch_normalization(intermediate,mean,variance,self.bias,self.gamma,self.epsion)
        else:
            self.hidden=intermediate+self.bias

        if self.dropout:
            self.hidden=tf.nn.dropout(self.hidden,keep_prob=self.keep_prob)

        if self.activation=='relu':
            self.layer=tf.nn.relu(self.hidden)
        elif self.activation=='tanh':
            self.layer=tf.nn.tanh(self.hidden)
        elif self.activation=='softmax':
            self.layer=tf.nn.softmax(self.hidden)
        elif self.activation=='none':
            self.layer=self.hidden

        return self.layer




