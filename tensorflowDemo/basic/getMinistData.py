from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("./tmp/data",one_hot=True)
print(mnist.train.num_examples)
print(mnist.validation.num_examples)
print(mnist.test.num_examples)
print(mnist.train.images[0])
print(mnist.train.labels[0])

batch_size=100
xs,ys=mnist.train.next_batch(batch_size)
print(xs.shape)
print(ys.shape)