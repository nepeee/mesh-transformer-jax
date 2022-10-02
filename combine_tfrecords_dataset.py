import tensorflow as tf
import glob
import random

files = glob.glob("*.tfrecords")
print(files)
dataset = tf.data.TFRecordDataset(files, num_parallel_reads=len(files))

filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(dataset)