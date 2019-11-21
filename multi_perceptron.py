import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def fc(x, weight, bias):
    return tf.add(tf.matmul(x, weight), bias)

def multilayer_perceptron(x, weight, bias):
    layer1 = fc(x, weight['h1'], bias['h1'])
    layer2 = fc(layer1, weight['h2'], bias['h2'])

    outlayer = tf.matmul(layer2, weight['out']) + bias['out']

    return outlayer

mnist = input_data.read_data_sets('/media/zhuzhu/ec114170-f406-444f-bee7-a3dc0a86cfa2/dataset/mnist', one_hot=True)

input_dims = 784
h1_dim = 256
h2_dim = 256
n_classes = 10
lr = 1e-3
num_epochs = 100
batch_size = 128
x = tf.placeholder(tf.float32, [None, input_dims])
y = tf.placeholder(tf.float32, [None, n_classes])
weight = {'h1': tf.Variable(tf.random_normal([input_dims, h1_dim])),
          'h2': tf.Variable(tf.random_normal([h1_dim, h2_dim])),
          'out': tf.Variable(tf.random_normal([h2_dim, n_classes]))}
bias = {'h1': tf.Variable(tf.zeros(h1_dim)),
        'h2': tf.Variable(tf.zeros(h2_dim)),
        'out': tf.Variable(tf.zeros(n_classes))}
logits = multilayer_perceptron(x, weight, bias)
loss_op = tf.nn.softmax_cross_entropy_with_logits(
    labels=y, logits=logits
)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        loss = 0.0
        for num_batch in range(int(mnist.train.num_examples / batch_size + 0.5)):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            _, running_loss = sess.run([train_op, loss_op], feed_dict={x: x_batch, y: y_batch})
            loss += running_loss.sum()
        loss /= mnist.train.num_examples
        print('Epoch:', '%03d'%(epoch+1), 'loss:', '%.3f'%loss)

    preds = tf.nn.softmax(logits)
    correct_predictions = tf.equal(tf.argmax(preds, axis=1), tf.argmax(y, axis=1))
    acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    print('Accuracy:', acc.eval({x: mnist.test.images, y: mnist.test.labels}))
