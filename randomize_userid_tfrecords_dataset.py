import tensorflow as tf
from transformers import AutoTokenizer
import random

tokenizer = AutoTokenizer.from_pretrained("twitch_j6_tokenizer")

def rand_userids(tokens):
    global tokenizer

    uidBase = 140
    idMinTok = 50257
    uidPrefs = [ 198, 50399, 31, 2488 ] #\n <|um|> @ @
    
    i = 2
    uidMap = {}
    while i < len(tokens)-2:
        if tokens[i] in uidPrefs:
            uid1 = tokens[i+1] - idMinTok
            uid2 = tokens[i+2] - idMinTok

            if (0 <= uid1 < uidBase) and (0 <= uid2 < uidBase):
                uidNum = uid1 * uidBase + uid2
                
                if not uidNum in uidMap:
                    while True:
                        newUidNum = random.randint(0, uidBase*uidBase -1)
                        if not newUidNum in uidMap:
                            break

                    uidMap[uidNum] = newUidNum
                else:
                    newUidNum = uidMap[uidNum]

                uid1, uid2 = divmod(newUidNum, uidBase)
                tokens[i+1] = uid1 + idMinTok
                tokens[i+2] = uid2 + idMinTok
                
                i += 3
                continue
        i += 1

    return tokens

raw_dataset = tf.data.TFRecordDataset("dataset_1351971.tfrecord")
features = {
    "text": tf.io.VarLenFeature(tf.int64)
}

ogText = open("ogText.txt", "w")
rndText = open("rndText.txt", "w")

for raw_record in raw_dataset.take(100):
	parsed_features = tf.io.parse_single_example(raw_record, features)
	data = tf.sparse.to_dense(tf.sparse.reorder(parsed_features["text"])).numpy()
	
	ogText.write(tokenizer.decode(data)+"\n\n")

	data = rand_userids(data)

	rndText.write(tokenizer.decode(data)+"\n\n")