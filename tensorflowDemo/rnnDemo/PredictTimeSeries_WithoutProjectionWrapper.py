import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def save_fig(fig_id, tight_layout=True):
    directory=os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


t_min,t_max=0,30
resolution=0.1

def time_series(t):
    return t*np.sin(t)/3+2*np.sin(5*t)

def next_batch(batch_size,n_steps):
    t0=np.random.rand(batch_size,1)*(t_max-t_min-(n_steps+1)*resolution)
    Ts=t0+np.arange(0.,n_steps+1)*resolution
    ys=time_series(Ts)
    return ys[:,:-1].reshape(-1,n_steps,1),ys[:,1:].reshape(-1,n_steps,1)

t=np.linspace(t_min,t_max,int((t_max-t_min)/resolution)+1)

'''
默认的情况下   endpoint=True  保留了最后一位,在画num个点的情况下,  step=(tmax-tmin)/(num-1)
endpoint=False 不保留最后一位,在画num个点的情况下,  step=(tmax-tmin)/(num)
'''

n_steps=20
t_instance=np.linspace(12.2,12.2+resolution*(n_steps),n_steps+1)

# plt.figure(figsize=(11,4))
# plt.subplot(121)
# plt.title("A time series (generated)", fontsize=14)
# plt.plot(t, time_series(t), label=r"$t * \sin(t) / 3 + 2 * \sin(5t)$")
# plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
# plt.legend(loc="lower left", fontsize=14)
# plt.axis([0, 30, -17, 13])
# plt.xlabel("Time")
# plt.ylabel("Value")
#
#
# plt.subplot(122)
# plt.title("A training instance", fontsize=14)
# plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=7, label="instance")
# plt.plot(t_instance[1:], time_series(t_instance[1:]), "r*", markersize=7, label="target")
# plt.legend(loc="upper left")
# plt.xlabel("Time")
# save_fig("time_series_plot")
# plt.show()


# X_batch,y_batch=next_batch(1,n_steps)
# print(X_batch.shape)
# print(np.c_[X_batch[0],y_batch[0]])


reset_graph()

n_inputs=1
n_neurons=100
n_outputs=1

X=tf.placeholder(tf.float32,shape=[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,shape=[None,n_steps,n_inputs])



# cell=tf.contrib.rnn.OutputProjectionWrapper(
#     tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu),
#     output_size=n_outputs
# )

cell=tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu)

outputs,states=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

stacked_rnn_outputs=tf.reshape(outputs,shape=[-1,n_neurons])

stacked_outputs=tf.layers.dense(stacked_rnn_outputs,n_outputs)
outputs=tf.reshape(stacked_outputs,[-1,n_steps,n_outputs])



#本地的模型恢复时，并不需要训练模型的阶段的代码。
learning_rate=0.001
loss=tf.reduce_mean(tf.square(outputs-y))

optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(loss)
n_iterations=1500
batch_size=50
init=tf.global_variables_initializer()

saver=tf.train.Saver()


# with tf.Session() as sess:
#     init.run()
#     for iteration in range(n_iterations):
#         X_batch,y_batch=next_batch(batch_size,n_steps)
#         sess.run(training_op,feed_dict={
#             X:X_batch,
#             y:y_batch
#         })
#         if iteration%100==0:
#             mse=loss.eval(feed_dict={X:X_batch,y:y_batch})
#             print(iteration,"\tMSE:",mse)
#
#     saver.save(sess,"./model/my_time_series_model_without_projection")

with tf.Session() as sess:
    saver.restore(sess, "./model/my_time_series_model_without_projection")  # not shown

    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})
    print(y_pred)

plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")

save_fig("time_series_pred_plot")
plt.show()







