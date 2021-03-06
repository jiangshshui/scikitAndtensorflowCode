import tensorflow as tf
import numpy as np

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()
n_steps=28
n_inputs=28
n_neurons=150
n_outputs=10

learning_rate=0.001

X=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.int32,shape=[None])

basic_cell=tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs,state=tf.nn.dynamic_rnn(basic_cell,X,dtype=tf.float32)

logits=tf.layers.dense(state,n_outputs)
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                             logits=logits)

loss=tf.reduce_mean(cross_entropy)
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(loss)

correct=tf.nn.in_top_k(logits,y,1)
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

init=tf.global_variables_initializer()

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("../tutorial/data/MNIST")
X_test=mnist.test.images.reshape((-1,n_steps,n_inputs))
y_test=mnist.test.labels


from datetime import datetime
now=datetime.now().strftime("%Y%m%d%H%M%S")
root_logdir="./tf_logs"
log_dir="{}/run-{}/".format(root_logdir,now)
loss_summary=tf.summary.scalar('LOSS',loss)
file_writer=tf.summary.FileWriter(logdir=log_dir,graph=tf.get_default_graph())

n_epochs=20
batch_size=200

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_iterations=int(np.ceil(mnist.train.num_examples//batch_size))
        for iteration in range(n_iterations):
            X_batch,y_batch=mnist.train.next_batch(batch_size)
            X_batch=X_batch.reshape((-1,n_steps,n_inputs))
            if iteration%10==0:
                summary_str=loss_summary.eval(feed_dict={X:X_batch,y:y_batch})
                step=epoch*(n_iterations)+iteration
                file_writer.add_summary(summary_str,step)
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})


        acc_train=accuracy.eval(feed_dict={X:X_batch,y:y_batch})
        acc_test=accuracy.eval(feed_dict={X:X_test,y:y_test})
        print("epoch:",epoch," Train Accuracy:",acc_train," Test Accuracy:",acc_test)

file_writer.close()
