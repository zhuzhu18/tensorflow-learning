import tensorflow as tf

x = tf.Variable([1, 2])
y = tf.constant([3, 3])

z = tf.subtract(x, y)
t = tf.add(z, y)

initializer = tf.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(initializer)
    r1 = sess.run(z)
    r2 = sess.run(t)
    print(r1)
    print(r2)
