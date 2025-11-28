# TikToken is a production-ready tokenizer used in OpenAI models,
# demonstrating how optimized subword vocabularies are applied in real systems.

import tiktoken

enc_gpt2 = tiktoken.get_encoding("r50k_base")
enc_gpt4 = tiktoken.get_encoding("cl100k_base")

sentence = "The 🌟 star-programmer implemented AGI overnight."

ids_gpt2 = enc_gpt2.encode(sentence)
ids_gpt4 = enc_gpt4.encode(sentence)

tokens_gpt2 = [enc_gpt2.decode([tid]) for tid in ids_gpt2]
tokens_gpt4 = [enc_gpt4.decode([tid]) for tid in ids_gpt4]

print("GPT-2:")
print(f"Vocabulary Size: {enc_gpt2.n_vocab} tokens")
print(f"Tokens (GPT-2): {tokens_gpt2}")
print(f"Ids (GPT-2): {ids_gpt2}")
print(f"Text (GPT-2): {enc_gpt2.decode(ids_gpt2)}\n")

print("GPT-4:")
print(f"Vocabulary Size: {enc_gpt4.n_vocab} tokens")
print(f"Tokens (GPT-4): {tokens_gpt4}")
print(f"Ids (GPT-4): {ids_gpt4}")
print(f"Text (GPT-4): {enc_gpt4.decode(ids_gpt4)}")