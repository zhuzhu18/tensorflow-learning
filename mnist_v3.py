import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def inference(input_tensor, avg_class, weight1, bias1,
              weight2, bias2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1)+bias1)
        layer2 = tf.matmul(layer1, weight2)+bias2

        return layer2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight1))+
                            avg_class.average(bias1))
        layer2 = tf.matmul(layer1, avg_class.average(weight2)) + avg_class.average(bias2)

        return layer2

def train(mnist, batch_size, feat_dim, hidden_dim, num_classes, moving_decay, reg, lr_base, lr_decay):
    x = tf.placeholder(tf.float32, [None, feat_dim])
    y = tf.placeholder(tf.float32, [None, num_classes])

    weight1 = tf.Variable(tf.truncated_normal([feat_dim, hidden_dim], mean=0, stddev=0.1))
    bias1 = tf.Variable(tf.constant(0.1, shape=[hidden_dim]))

    weight2 = tf.Variable(tf.truncated_normal([hidden_dim, num_classes], stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape=[num_classes]))

    target = inference(x, None, weight1, bias1, weight2, bias2)
    global_step = tf.Variable(0, trainable=False)

    avg_class = tf.train.ExponentialMovingAverage(moving_decay, global_step)
    variables_average_op = avg_class.apply(tf.trainable_variables())
    average_y = inference(x, avg_class, weight1, bias1, weight2, bias2)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=average_y, labels=tf.argmax(y, 1))
    regularizer = tf.contrib.layers.l2_regularizer(reg)

    total_loss = tf.reduce_mean(loss) + regularizer(weight1) + regularizer(weight2)
    lr = tf.train.exponential_decay(lr_base, global_step, decay_steps=mnist.train.num_examples / batch_size, decay_rate=lr_decay)

    train_step = tf.train.GradientDescentOptimizer(lr).minimize(total_loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_average_op]):
        tf.no_op(name='train')
    correct = tf.equal(tf.argmax(average_y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(10000):
            for i in range(int(mnist.train.num_examples / batch_size)):
                xs, ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: xs, y: ys})
            if step % 100 == 0:
                print('After %i training steps, validation accuracy: '%step, sess.run(accuracy,
                        feed_dict={x: mnist.validation.images, y: mnist.validation.labels}))

def main(argv=None):
    mnist = input_data.read_data_sets('/media/zhuzhu/ec114170-f406-444f-bee7-a3dc0a86cfa2/dataset/MNIST/raw',
                                      one_hot=True)
    train(mnist, 100, 784, 500, 10, 0.99, 1e-4, 0.8, 0.99)

if __name__ == '__main__':
    tf.app.run()
