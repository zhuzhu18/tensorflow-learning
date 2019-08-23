import tensorflow as tf
import os
from numpy.random import RandomState
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1))

x = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
y = tf.sigmoid(y)
criterion = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0))+
                             (1-y_)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(criterion)

rdm = RandomState(1)
datasize = 128
X = rdm.rand(datasize, 2)
Y = [[int((x1+x2)<1)] for x1, x2 in X]
batchsize = 8

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print('before training, w1=', sess.run(w1))
    print('before training, w2=', w2.eval())

    steps = 5000
    for step in range(steps):
        start = (step * batchsize) % datasize
        end = min(start+batchsize, datasize)

        sess.run(train_step, feed_dict={x:X[start:end],y_:Y[start:end]})
        if step % 100 == 0:
            loss = sess.run(criterion, feed_dict={x:X, y_:Y})
            print("After %d training steps, crossentropy on all data is %g"%(step, loss))
    print('after training, w1=', sess.run(w1))
    print('after training, w2=', sess.run(w2))
