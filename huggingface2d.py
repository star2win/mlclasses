import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
batched_ids = [ids, ids]

input_ids = tf.constant(batched_ids)
print("Input IDs:", input_ids)
print("\n\n\n")

output = model(input_ids)
print("Logits:", output.logits)


sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids2 = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

print("\n\n\n")
print(model(tf.constant(sequence1_ids)).logits)
print("\n\n\n")
print(model(tf.constant(sequence2_ids)).logits)
print("\n\n\n")
print(model(tf.constant(batched_ids2), attention_mask=tf.constant(attention_mask)).logits)


#tokenized_inputs = tokenizer(sequence, return_tensors="tf")
#print(tokenized_inputs["input_ids"])

# This line will fail.
#model(input_ids)


