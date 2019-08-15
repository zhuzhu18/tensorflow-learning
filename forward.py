import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os
# import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y), _ = datasets.mnist.load_data()
# x: [60k, 28, 28]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y ,dtype=tf.int32)

train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(32)
train_iter = iter(train_db)
sample = next(train_iter)

# [b, 784] ==> [784, 256] ==> [256, 128] ==> [128, 10]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3
for epoch in range(10):
    for step, (x, y) in enumerate(train_iter):
        x = tf.reshape(x, [-1, 28 * 28])
        y = tf.one_hot(y, depth=10)
    
        with tf.GradientTape() as tape:
            h1 = tf.nn.relu(x @ w1 + tf.broadcast_to(b1, [x.shape[0], 256]))
            h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
            h3 = tf.matmul(h2, w3) + b3
    
            loss = tf.reduce_mean(tf.square(y - h3))
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        w1 = w1.assign_sub(lr*grads[0])
        b1 = b1.assign_sub(lr*grads[1])
        w2 = w2.assign_sub(lr*grads[2])
        b2 = b2.assign_sub(lr*grads[3])
        w3 = w3.assign_sub(lr*grads[4])
        b3 = b3.assign_sub(lr*grads[5])
    
        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))
