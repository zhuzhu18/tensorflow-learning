import tensorflow as tf
import numpy as np


x = np.random.rand(100)
y = 0.1*x+0.2

k = tf.Variable(0.)
b = tf.Variable(0.)

m = k*x+b
init = tf.initialize_all_variables()
loss = tf.reduce_mean(tf.square(m-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(init)
    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run([k, b]))

