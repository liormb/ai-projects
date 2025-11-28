# Subword/BPE balances between Word-level and Character-level, 
# and is the default choice for most modern LLMs.

from transformers import AutoTokenizer

sample = "Unbelievable tokenization powers! 🚀"

class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")

    def size(self):
        return self.tokenizer.vocab_size

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

tokenizer = Tokenizer()
ids = tokenizer.encode(sample)
text = tokenizer.decode(ids)

print(f"Vocabulary Size: {tokenizer.size()} tokens")
print(f"Tokens: {tokenizer.tokenize(sample)}")
print(f"Ids: {ids}")
print(f"Text: {tokenizer.decode(ids)}")
