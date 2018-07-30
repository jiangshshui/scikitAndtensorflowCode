import numpy as np
import tensorflow as tf

class PoolLayer:
    def __init__(self,pool_size,stride,mode='max',padding='SAME',
                 resp_normal=False,name='pool'):
        self.pool_size=pool_size
        self.stride=stride
        self.mode=mode
        self.padding=padding
        self.resp_normal=resp_normal

    def get_layer(self,input):
        if self.mode=='max':
            self.pool=tf.nn.max_pool(
                value=input,
                ksize=[1,self.pool_size,self.pool_size,1],
                strides=[1,self.stride,self.stride,1],
                padding=self.padding
            )
        elif self.mode=='avg':
            self.pool=tf.nn.avg_pool(value=input,
                                     ksize=[1,self.pool_size,self.pool_size,1],
                                     strides=[1,self.stride,self.stride,1],
                                     padding=self.padding)

        self.output_shape=self.pool.get_shape().as_lsit()
        if self.resp_normal:
            self.hidden=tf.nn.local_response_normalization(self.pool,depth_radius=7,alpha=0.001,beta=0.75)
        else:
            self.hidden=self.pool

        return self.pool