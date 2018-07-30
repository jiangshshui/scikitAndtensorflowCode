import tensorflow as tf
import numpy as np

n_steps=2
n_inputs=3
n_neurons=5

X=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
X_transpose=tf.transpose(X,perm=[1,0,2])
X_seqs=tf.unstack(X_transpose)
basic_cell=tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)

output_seqs,states=tf.contrib.rnn.static_rnn(basic_cell,X_seqs,
                                             dtype=tf.float32)
outputs=tf.transpose(tf.stack(output_seqs),perm=[1,0,2])

X_batch=np.array([
    [[0,1,2],[9,8,7]],
    [[3,4,5],[0,0,0]],
    [[6,7,8],[6,5,4]],
    [[9,0,1],[3,2,1]]
    ])
print(X_batch.shape)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    #sess.run(X_seqs,feed_dict={X:X_batch})
    #output_val = outputs.eval(feed_dict={X: X_batch})#   输入的tensor 的错误!!!

    # output_val=outputs.eval(feed_dict={X:X_batch})
    # print(X_transpose.eval(feed_dict={X:X_batch}))
    # print(X_seqs[0].eval(feed_dict={X:X_batch}))

    X_transpose_r,X_seqs_r,outputs_r=sess.run([X_transpose,X_seqs[0],outputs],feed_dict={X:X_batch})
    print(X_transpose_r)
    '''
    [[[0. 1. 2.]
  [3. 4. 5.]
  [6. 7. 8.]
  [9. 0. 1.]]

 [[9. 8. 7.]
  [0. 0. 0.]
  [6. 5. 4.]
  [3. 2. 1.]]]
    '''
    print(X_seqs_r)
    '''
    [[0. 1. 2.]
    [3. 4. 5.]
    [6. 7. 8.]
    [9. 0. 1.]]
    '''
    print(outputs_r)


