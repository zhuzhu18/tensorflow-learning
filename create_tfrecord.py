import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
mnist = input_data.read_data_sets('/media/tl/213/data/mnist', dtype=tf.uint8, one_hot=True)

images = mnist.train.images
labels = mnist.train.labels

num_examples = mnist.train.num_examples
img_size = images.shape[1]
filename = '/media/tl/213/data/mnist/mnist_records.tfrecords'
writer = tf.io.TFRecordWriter(filename)
for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'img_size': _int64_feature(img_size),
            'label': _int64_feature(np.argmax(labels[index])),
            'image_raw': _bytes_feature(image_raw)
        }
    ))
    writer.write(example.SerializeToString())
writer.close()
