import tensorflow as tf
import numpy as np

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


reset_graph()
n_inputs=28
n_steps=28
n_outputs=10

learning_rate=0.001

X=tf.placeholder(tf.float32,shape=[None,n_steps,n_inputs])
y=tf.placeholder(tf.int32,shape=[None])

n_neurons=100
n_layers=3

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../tutorial/data/MNIST")
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

layers=[tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu)
            for layer in range(n_layers)]

multi_layer_cell=tf.contrib.rnn.MultiRNNCell(layers)

outputs,states=tf.nn.dynamic_rnn(multi_layer_cell,X,dtype=tf.float32)

states_concat=tf.concat(axis=1,values=states)
logits=tf.layers.dense(states_concat,n_outputs)

cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)

loss=tf.reduce_mean(cross_entropy)

optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(loss)

correct=tf.nn.in_top_k(logits,y,1)
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

init=tf.global_variables_initializer()

n_epochs=20
batch_size=150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(int(np.ceil(mnist.train.num_examples // batch_size))):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)





