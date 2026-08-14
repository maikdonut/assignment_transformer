import argparse
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from baseline import run_baseline
from embeddings import get_cls_embeddings, load_encoder
from tokenization import load_tokenizer, tokenize_texts

PLOTS_DIR = Path("attention_plots")


def similarity(text1, text2, tokenizer, model):
    emb = get_cls_embeddings([text1, text2], tokenizer=tokenizer, model=model)
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


def smoke_tokenization():
    print("=== Задача 1: токенизация ===")
    tokenizer = load_tokenizer()
    texts = [
        "This movie was great!",
        "Terrible movie, waste of time.",
        "I have mixed feelings about this game.",
    ]
    tokens = tokenize_texts(texts, tokenizer=tokenizer)
    print(f"input_ids shape: {tokens['input_ids'].shape}")
    print(f"attention_mask:\n{tokens['attention_mask']}")
    for i, text in enumerate(texts):
        decoded = tokenizer.convert_ids_to_tokens(tokens["input_ids"][i])
        print(f"\nТекст: {text}")
        print(f"Токены: {decoded}")
    return tokens


def smoke_embeddings():
    print("\n=== Задача 2: CLS-эмбеддинги ===")
    texts = [
        "This movie was great!",
        "Terrible movie, waste of time.",
        "I have mixed feelings about this game.",
    ]
    embeddings = get_cls_embeddings(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    expected = (len(texts), 768)
    if embeddings.shape != expected:
        raise ValueError(f"Ожидали shape {expected}, получили {embeddings.shape}")
    return embeddings


def run_attention():
    tokenizer, model, device = load_encoder(output_attentions=True)

    text = "The amazing movie won many awards"
    tokens = tokenizer(text, return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}
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
    sentiment_tokens = {key: value.to(device) for key, value in sentiment_tokens.items()}
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


def parse_args():
    parser = argparse.ArgumentParser(description="Assignment transformer: baseline и attention")
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Запустить визуализацию внимания (День 3)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.attention:
        run_attention()
        return

    smoke_tokenization()
    smoke_embeddings()
    print("\n=== Задача 3: Logistic Regression ===")
    run_baseline()


if __name__ == "__main__":
    main()
