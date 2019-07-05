import tensorflow as tf
from tensorflow.examples.tutorials import mnist

mnist_data = mnist.input_data.read_data_sets('mnist_data', one_hot=True)

batch_size = 100
num_batch = mnist_data.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

weight = tf.Variable(tf.zeros((784, 10), tf.float32))
bias = tf.Variable(tf.zeros(10, tf.float32))

predict = tf.nn.softmax(tf.matmul(x, weight) + bias)

loss = tf.reduce_mean(tf.square(y - predict))
correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(predict, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(20):
        for step in range(num_batch):
            data, label = mnist_data.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:data, y:label})
        acc = sess.run(accuracy, feed_dict={x:mnist_data.test.images, y:mnist_data.test.labels})
        print('epoch {}, accuracy {}'.format(epoch, acc))

