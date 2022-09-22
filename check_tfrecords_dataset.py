import tensorflow as tf
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("twitch_j6_tokenizer")

raw_dataset = tf.data.TFRecordDataset("out/dataset_337953.tfrecords")
features = {
    "text": tf.io.VarLenFeature(tf.int64)
}

for raw_record in raw_dataset.take(100):
	#example = tf.train.Example()
	#example.ParseFromString(raw_record.numpy())

	parsed_features = tf.io.parse_single_example(raw_record, features)
	data = tf.sparse.to_dense(tf.sparse.reorder(parsed_features["text"])).numpy() #tf.cast(tf.sparse.to_dense(tf.sparse.reorder(parsed_features["text"])), tf.uint32)
	
	print(tokenizer.decode(data))
	input()