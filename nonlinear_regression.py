import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5, 0.5, 200, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

weight1 = tf.Variable(tf.random_normal([1, 10]))

bias1 = tf.Variable(tf.zeros(1, 1))
weight2 = tf.Variable(tf.random_normal([10, 1]))
bias2 = tf.Variable(tf.zeros(1, 1))

l1 = tf.nn.tanh(tf.matmul(x_data, weight1) + bias1)
l2 = tf.nn.tanh(tf.matmul(l1, weight2) + bias2)
loss = tf.reduce_mean(tf.square(y_data - l2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        sess.run(train, feed_dict={x: x_data, y: y_data})
    predict = sess.run(l2, feed_dict={x: x_data})

plt.figure()
plt.scatter(x_data, y_data)
plt.plot(x_data, predict, 'r-')
plt.show()
