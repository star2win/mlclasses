import os

# Disable parallelism to avoid the warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import BertTokenizer
from transformers import AutoTokenizer

tokenizer1 = BertTokenizer.from_pretrained("bert-base-cased")
tokenizer2 = AutoTokenizer.from_pretrained("bert-base-cased")

sequence1 = "Using a Transformer network is simple"
output1 = tokenizer1(sequence1)

sequence2 = "I've been waiting for a HuggingFace coruse my whole life."
output2 = tokenizer2(sequence2)

#tokenizer1.save_pretrained("./temp")

tokens1 = tokenizer1.tokenize(sequence1)
tokens2 = tokenizer1.tokenize(sequence2)

ids1 = tokenizer1.convert_tokens_to_ids(tokens1)
ids2 = tokenizer1.convert_tokens_to_ids(tokens2)

decoded_string2 = tokenizer2.decode(ids2)

print("\n\n\n")
print(output2)

print("\n\n\n")
print(tokens2)

print("\n\n\n")
print(ids2)

print("\n\n\n")
print(decoded_string2)