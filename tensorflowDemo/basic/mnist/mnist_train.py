import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflowDemo.basic.mnist import mnist_inference

BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99

MODEL_SAVE_PATH="./save/"
MODEL_NAME="mnistmodel.ckpt"

def train(mnist):
    x=tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name="x-input")
    y_label=tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name="y-input")

    # regularizer=tf.contrib.layers.l2_reaularizer[REGULARIZATION_RATE]
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y_output=mnist_inference.inference(x,regularizer)

    global_step=tf.Variable(0,trainable=False)
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_averages_op=variable_averages.apply(tf.trainable_variables())
    # print(tf.shape(y_label))
    # print(tf.shape(y_output))
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_label,1),logits=y_output)
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection("losses"))
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op=tf.no_op(name="train")

    init_op=tf.global_variables_initializer()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(TRAINING_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_label:ys})

            if i%1000==0:
                print("After %d training step(s), loss on training batch is %g"%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv=None):
    mnist=input_data.read_data_sets("./tmp/data",one_hot=True)
    train(mnist)

if __name__=="__main__":
    tf.app.run()






