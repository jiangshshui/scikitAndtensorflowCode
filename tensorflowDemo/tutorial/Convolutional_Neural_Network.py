import tensorflow as tf
import time
import numpy as np
import datetime
from tensorflow.examples.tutorials.mnist import input_data
data=input_data.read_data_sets('./data/MNIST',one_hot=True)
data.test.cls=np.argmax(data.test.labels,axis=1)
filter_size1=5
#num_filters1=15
num_filters1=20
filter_size2=3
num_filters2=36
fc_size=128
img_size=28
img_size_flatten=img_size*img_size
img_shape=(img_size,img_size)
num_channels=1
num_classes=10

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.5))

def new_biases(length):
    return tf.constant(0.05,dtype=tf.float32,shape=[length])

def new_conv_layer(input,
                   filter_size,
                   num_input_channels,
                   num_output_channels,
                   use_pooling=False):
    shape=[filter_size,filter_size,num_input_channels,num_output_channels]
    weights=new_weights(shape)#filter 的维度
    biases=new_biases(num_output_channels)
    layer=tf.nn.conv2d(input=input,
                       filter=weights,
                       strides=[1,1,1,1],
                       padding='SAME')
    layer+=biases
    if use_pooling:
        layer=tf.nn.max_pool(value=layer,
                             ksize=[1,2,2,1],
                             strides=[1,2,2,1],
                             padding='SAME')
    return layer,weights


def flatten_layer(layer):
    layer_shape=layer.get_shape()
    num_features=layer_shape[1:4].num_elements()
    layer_flat=tf.reshape(layer,[-1,num_features])
    return layer_flat,num_features


def new_fc_layer(input,
                 num_inputs,
                 num_outputs,
                 use_relu=True):
    weights=new_weights(shape=[num_inputs,num_outputs])
    biases=new_biases(length=num_outputs)
    layer=tf.matmul(input,weights)+biases
    if use_relu:
        layer=tf.nn.relu(layer)
    return layer

x=tf.placeholder(tf.float32,shape=[None,img_size_flatten],name='x')
x_image=tf.reshape(x,[-1,img_size,img_size,num_channels])
y_true=tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')
y_true_cls=tf.argmax(y_true,axis=1)

layer_conv1,weights_conv1=new_conv_layer(input=x_image,
                                        num_input_channels=num_channels,
                                        filter_size=filter_size1,
                                        num_output_channels=num_filters1,
                                        use_pooling=True)

layer_conv2,weights_conv2=new_conv_layer(input=layer_conv1,
                                         num_input_channels=num_filters1,
                                         filter_size=filter_size2,
                                         num_output_channels=num_filters2,
                                         use_pooling=True)

layer_flat,num_features=flatten_layer(layer_conv2)


layer_fc1=new_fc_layer(input=layer_flat,
                       num_inputs=num_features,
                       num_outputs=fc_size,
                       use_relu=True)


layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)  #最后一层不要relu


y_pred=tf.nn.softmax(layer_fc2)
y_pred_cls=tf.argmax(y_pred,axis=1)
cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,labels=y_true)
cost=tf.reduce_mean(cross_entropy)
#optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_precdition=tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct_precdition,tf.float32))
train_batch_size=64
total_iteration=0
sess=tf.Session()
sess.run(tf.global_variables_initializer())



total_iteration=0
def optimize(num_iterations):
    global total_iteration
    start_time=time.time()
    for i in range(total_iteration,total_iteration+num_iterations):
        x_batch,y_true_batch=data.train.next_batch(train_batch_size)
        feed_dict_train={
            x:x_batch,
            y_true:y_true_batch
        }

        sess.run(optimizer,feed_dict=feed_dict_train)
        if i%100==0:
            acc=sess.run(accuracy,feed_dict=feed_dict_train)
            msg="Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.3%}"
            #https://www.cnblogs.com/nulige/p/6115793.html  格式化
            print(msg.format(i+1,acc))

    total_iteration+=num_iterations
    end_time=time.time()

    time_dif=end_time-start_time
    print("Time usage: "+str(datetime.timedelta(seconds=int(round(time_dif)))))




test_batch_size=256

def print_test_accuracy():
    num_test=len(data.test.images)
    cls_pred=np.zeros(shape=num_test,dtype=np.int)
    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]
        feed_dict = {x: images,
                     y_true: labels}
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    cls_true = data.test.cls
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.3%} ({1} / {2})"
    print(msg.format(acc,correct_sum,num_test))

optimize(10000)
print_test_accuracy()

sess.close()
