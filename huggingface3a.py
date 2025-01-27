import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = [
    "i've been waiting for a HuggingFace course my whole life.",
    "This course is amzing!",
]

batch = dict(tokenizer(sequences, padding=True, truncation=True, return_tensors="tf"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
labels = tf.convert_to_tensor([1,1])
model.train_on_batch(batch, labels)

