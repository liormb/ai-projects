import torch, transformers, tiktoken

corpus = [
    "The quick brown fox jumps over the lazy dog",
    "Tokenization converts text to numbers",
    "Large language models predict the next token"
]

class Tokenizer:
    def __init__(self, corpus):
        self.vocab = []
        self.word2id = {}
        self.id2word = {}
        self.UKN_TOKEN = "<UNK>"
        self.build_vocab(corpus)


    def build_vocab(self, corpus):
        for text in corpus:
            self.add_tokens_to_vocab(text)
        return len(self.vocab)

    def add_tokens_to_vocab(self, text):
        for token in text.split():
            if token not in self.vocab:
                self.vocab.append(token)
                self.word2id[token] = len(self.vocab) - 1
                self.id2word[len(self.vocab) - 1] = token

    def size(self):
        return len(self.vocab)
    
    def encode(self, text):
        self.add_tokens_to_vocab(text)
        return [self.word2id[token] for token in text.split()]

    def decode(self, ids):
        return ' '.join([self.id2word[id] if id in self.id2word else self.UKN_TOKEN for id in ids])
    
tokenizer = Tokenizer(corpus)

sentence = "The text tokenization used in language model is like a quick fox caching the next lazy dog"
print(f"Vocabulary Size: {tokenizer.size()} tokens")
print(f"Tokens: {tokenizer.vocab}\n")

# Test encoding by adding new words from the sentence
print(f"Adding tokens to vocabulary: \n{sentence}\n")

ids = tokenizer.encode(sentence)

print(f"Vocabulary Size: {tokenizer.size()} tokens")
print(f"New tokens: {tokenizer.vocab}\n")

print(f"Ids: {ids}")

# Test with 9999 as an unknown token ID
print(f"Text: {tokenizer.decode([*ids, 9999])}")