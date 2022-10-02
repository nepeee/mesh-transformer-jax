import tensorflow as tf
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("twitch_j6_tokenizer")

raw_dataset = tf.data.TFRecordDataset("dataset_336085.tfrecords")
features = {
    "text": tf.io.VarLenFeature(tf.int64)
}

for raw_record in raw_dataset.take(100):
	parsed_features = tf.io.parse_single_example(raw_record, features)
	data = tf.sparse.to_dense(tf.sparse.reorder(parsed_features["text"])).numpy()
	
	print(tokenizer.decode(data))
	input()