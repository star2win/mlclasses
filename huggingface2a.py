from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="tf")
outputs = model(inputs)

print("\n\n\n")

print(inputs)

print("\n\n\n")

outputs = model(inputs)

print(outputs.logits.shape)

print("\n\n\n")

print(outputs.logits)

#print(outputs.last_hidden_state.shape)

print("\n\n\n")

predictions = tf.math.softmax(outputs.logits, axis=-1)
print(predictions)


