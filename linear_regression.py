import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()

train_x = np.linspace(0, 10, 100) + np.random.rand(100)
train_y = np.linspace(2, 12, 100)

w = tf.Variable(np.random.randn(), name='weight')
b = tf.Variable(np.random.randn(), name='bias')

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
pred = tf.multiply(x, w) + b
loss = tf.reduce_mean(tf.square(y-pred))
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1000):
        for x_, y_ in zip(train_x, train_y):
            sess.run(train_step, feed_dict={x:x_, y:y_})
            if epoch % 100 == 0:
                print('training loss = %f, w = %f, b = %f'%(sess.run(loss, feed_dict={x:x_, y:y_}), sess.run(w), sess.run(b)))
    plt.plot(train_x, train_y, 'bo')
    sample = np.linspace(train_x.min(), train_x.max(), 1000)
    plt.plot(sample, sess.run(w)*sample+sess.run(b))
plt.show()
