import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/media/zhuzhu/0C5809B80C5809B8/常用数据集/MNIST/raw', one_hot=True)
batch_size = 32

n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

w1 = tf.Variable(initial_value=tf.truncated_normal([784, 100], stddev=0.1))
b1 = tf.Variable(initial_value=tf.zeros([100])+0.1)
l1 = tf.nn.tanh(tf.matmul(x, w1)+b1)
l1_drop = tf.nn.dropout(l1, keep_prob)

w2 = tf.Variable(initial_value=tf.truncated_normal([100, 100], stddev=0.1))
b2 = tf.Variable(initial_value=tf.zeros([100])+0.1)
l2 = tf.nn.tanh(tf.matmul(l1_drop, w2)+b2)
l2_drop = tf.nn.dropout(l2, keep_prob)

w3 = tf.Variable(initial_value=tf.truncated_normal([100, 10], stddev=0.1))
b3 = tf.Variable(initial_value=tf.zeros([10])+0.1)
l3 = tf.matmul(l2_drop, w3)+b3
prediction = tf.nn.softmax(l3)


loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(20):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_x,y:batch_y,keep_prob:1.0})
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        print('Iter '+str(epoch)+',Testing Accuracy '+str(acc))
