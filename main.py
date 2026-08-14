from pathlib import Path

from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

PLOTS_DIR = Path("attention_plots")

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


def visualize_attention(tokens, attention, tokenizer, layer=0, head=0, suffix=""):
    """
    tokens: токенизированный текст
    attention: attention weights от модели
    tokenizer: токенизатор для подписей осей
    layer: номер слоя для визуализации
    head: номер головы для визуализации
    suffix: суффикс имени файла, чтобы не перезаписывать графики
    """
    attn = attention[layer][0, head]
    token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    PLOTS_DIR.mkdir(exist_ok=True)
    filename = f"attention_layer{layer}_head{head}{suffix}.png"
    filepath = PLOTS_DIR / filename

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn.cpu().numpy(),
        xticklabels=token_list,
        yticklabels=token_list,
        cmap="viridis",
        cbar=True,
    )
    plt.title(f"Attention - Layer {layer}, Head {head}")
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    print(f"Сохранено: {filepath}")
    print(f"Токены: {token_list}")
    return filepath


def top_attended_tokens(tokens, attention, tokenizer, query_token, layer=5, head=0, top_k=5):
    """Печатает токены с наибольшим вниманием для query-позиции query_token."""
    token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    query_positions = [i for i, tok in enumerate(token_list) if tok == query_token]
    if not query_positions:
        print(f'Токен "{query_token}" не найден среди {token_list}')
        return

    attn = attention[layer][0, head]
    skip = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}

    for pos in query_positions:
        weights = attn[pos]
        ranked = sorted(
            enumerate(weights.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        ranked = [(i, w) for i, w in ranked if token_list[i] not in skip][:top_k]
        print(f'\nQuery "{token_list[pos]}" (позиция {pos}), слой {layer}, голова {head}:')
        for i, weight in ranked:
            print(f"  {token_list[i]:12s}  {weight:.4f}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(EN_MODEL_NAME)
    model = AutoModel.from_pretrained(EN_MODEL_NAME, output_attentions=True)
    model.eval()

    text = "The amazing movie won many awards"
    tokens = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)

    print(type(outputs.attentions))
    print(f"Количество слоёв: {len(outputs.attentions)}")
    print(f"Форма attention для слоя 0: {outputs.attentions[0].shape}")

    attention = outputs.attentions[0]
    print(f"Attention shape: {attention.shape}")
    attn_single = attention[0, 0]
    print(f"Single head shape: {attn_single.shape}")

    last_layer = len(outputs.attentions) - 1
    mid_layer = len(outputs.attentions) // 2
    print("\n--- Слои (голова 0) ---")
    visualize_attention(tokens, outputs.attentions, tokenizer, layer=0, head=0)
    visualize_attention(tokens, outputs.attentions, tokenizer, layer=mid_layer, head=0)
    visualize_attention(tokens, outputs.attentions, tokenizer, layer=last_layer, head=0)

    print("\n--- Головы слоя 0 ---")
    for head in range(model.config.num_attention_heads):
        visualize_attention(tokens, outputs.attentions, tokenizer, layer=0, head=head)

    sentiment_text = "This movie was absolutely terrible and I hated it"
    sentiment_tokens = tokenizer(sentiment_text, return_tensors="pt")
    with torch.no_grad():
        sentiment_outputs = model(**sentiment_tokens)

    print("\n--- Сентимент-пример ---")
    visualize_attention(
        sentiment_tokens,
        sentiment_outputs.attentions,
        tokenizer,
        layer=last_layer,
        head=0,
        suffix="_sentiment",
    )
    top_attended_tokens(
        sentiment_tokens,
        sentiment_outputs.attentions,
        tokenizer,
        query_token="terrible",
        layer=last_layer,
        head=0,
    )

if __name__ == "__main__":
    main()
