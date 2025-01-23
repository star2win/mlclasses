from transformers import BertConfig, TFBertModel
from transformers import AutoTokenizer
import tensorflow as tf

# Building the config
#config = BertConfig()
# Building the model from the config
#model = TFBertModel(config)
# print(config)

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFBertModel.from_pretrained(checkpoint)

#model.save_pretrained("./temp")

sequences = ["Hello!", "Cool.", "Nice!"]
inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="tf")

#encoded_sequences = inputs['input_ids']
#encoded_sequences = [
#    [101, 7592, 999, 102],
#    [101, 4658, 1012, 102],
#    [101, 3835, 999, 102],]

#model_inputs = tf.constant(encoded_sequences)

output = model(inputs)

print("\n\n\n")
print(output)
