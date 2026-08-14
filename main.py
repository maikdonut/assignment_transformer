from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
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


def get_embeddings(texts, tokenizer, model, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenize_texts(batch_texts, tokenizer)

        with torch.no_grad():
            outputs = model(**tokens)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def similarity(text1, text2, tokenizer, model):
    emb = get_embeddings([text1, text2], tokenizer, model)
    sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
    return sim


def main():
    tokenizer = AutoTokenizer.from_pretrained(EN_MODEL_NAME)
    explain_tokenization("Transformers are amazing!", tokenizer)

    model = AutoModel.from_pretrained(EN_MODEL_NAME)
    model.eval()
    print(model)

    text = "This movie was absolutely amazing!"
    tokens = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)

    print(type(outputs))
    print(outputs.last_hidden_state.shape)

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    print(f'CLS embedding shape: {cls_embedding.shape}')
    print(f'CLS embedding: {cls_embedding[0][:5]}...')

    texts = [
        "This movie was absolutely amazing!",
        "Terrible movie, waste of time.",
        "Pretty good, I liked it.",
        "Boring and too long."
    ]
    embeddings = get_embeddings(texts, tokenizer, model)
    print(f'Embeddings shape: {embeddings.shape}')
    print('Ожидается: (4, 768) для DistilBERT')

    sim1 = similarity("Great movie!", "Amazing film!", tokenizer, model)
    sim2 = similarity("Great movie!", "Terrible film!", tokenizer, model)
    print(f'Сходство похожих: {sim1:.3f}')
    print(f'Сходство разных: {sim2:.3f}')

if __name__ == "__main__":
    main()
