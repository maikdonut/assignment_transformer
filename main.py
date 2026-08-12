from transformers import AutoTokenizer
import torch

EN_MODEL_NAME = "distilbert-base-uncased"
RU_MODEL_NAME = "distilbert-base-multilingual-cased"

def tokenize_texts(texts, tokenizer, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def explain_tokenization(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f'Исходный текст: {text}')
    print(f'Токены: {tokens}')
    print(f'IDs: {ids}')
    print(f'Количество: {len(tokens)}')



# print(tokenizer.vocab_size)
# print(tokenizer.model_max_length)
# text = "This movie was absolutely amazing!"
# tokens = tokenizer(text)
# print(tokens)

# input_ids = tokens["input_ids"]
# print(tokenizer.convert_ids_to_tokens(input_ids))
# print(f"Количество токенов: {len(input_ids)}")
# decoded = tokenizer.decode(input_ids)
# print(f"Декодировано: {decoded}")



# texts = [
#     "This movie was great!",
#     "Terrible movie, waste of time."
# ]

# tokens = tokenize_texts(texts, tokenizer)
# print(f'Shape: {tokens["input_ids"].shape}')
# print(f'Attention mask:\n{tokens["attention_mask"]}')
# print(tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]))
# print(tokenizer.convert_ids_to_tokens(tokens["input_ids"][1]))


# print(f'CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})')
# print(f'SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})')
# print(f'PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})')

# single = tokenizer(text, return_tensors="pt")
# print(f'Input IDs: {single["input_ids"]}')
# print(f'Decoded: {tokenizer.decode(single["input_ids"][0])}')


def main():
    tokenizer = AutoTokenizer.from_pretrained(EN_MODEL_NAME)
    explain_tokenization("Transformers are amazing!", tokenizer)

if __name__ == "__main__":
    main()
