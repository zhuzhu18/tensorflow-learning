import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/media/zhuzhu/ec114170-f406-444f-bee7-a3dc0a86cfa2/dataset/mnist')
num_examples = mnist.train.num_examples
images = mnist.train.images    # numpy float32类型, 50000 * 784
labels = mnist.train.labels    # numpy uint8类型, 50000

file_path = './a/mnist_train.tfrecords'
writer = tf.io.TFRecordWriter(path=file_path)

for i in range(num_examples):
    images_feature = tf.train.Feature(float_list=tf.train.FloatList(value=images[i].tolist()))
    labels_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i].tolist()]))

    example = tf.train.Example(features=tf.train.Features(feature={
        'images': images_feature,
        'labels': labels_feature
    }))
    writer.write(example.SerializeToString())

writer.close()
