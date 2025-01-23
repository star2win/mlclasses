import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
sequence = ["I've been waiting for HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequence, padding=True, truncation=True, return_tensors="tf")
output = model(**tokens)

print("\n\n\n")
print(output)