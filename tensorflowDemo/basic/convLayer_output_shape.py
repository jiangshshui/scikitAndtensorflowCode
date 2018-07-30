import tensorflow as tf
import numpy as np


def generate_image(image_size,numbers):
    #return np.random.rand(numbers,image_size*image_size)
    return tf.random_normal(shape=[numbers,image_size*image_size])


def new_convlayer(inputs,input_channels,output_channels,filter_size,stride,padding='SAME'):
    input_shape=inputs.get_shape().as_list()

    weights=tf.Variable(initial_value=tf.truncated_normal(shape=[filter_size,filter_size,input_channels,output_channels],
                                                          mean=0,
                                                          stddev=np.sqrt(2.0/(input_shape[1]*input_shape[2]*input_shape[3])),
                                                          dtype=tf.float32),
                        dtype=tf.float32)

    conv_layer=tf.nn.conv2d(input=inputs,filter=weights,strides=stride,padding=padding)

    return conv_layer


if __name__=='__main__':
    img_size=28
    filter_size=5
    stride=[1,5,5,1]
    #padding="SAME"
    padding="VALID"
    images=generate_image(img_size,10)
    #inputs=images.reshape([-1,img_size,img_size,1])
    inputs=tf.reshape(images,shape=[-1,img_size,img_size,1])
    layer=new_convlayer(inputs=inputs,input_channels=1,output_channels=5,filter_size=filter_size,stride=stride,padding=padding)
    print(layer.get_shape())





