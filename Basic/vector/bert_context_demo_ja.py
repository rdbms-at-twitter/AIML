# -*- coding: utf-8 -*-
"""
日本語 BERT 文脈依存ベクトル 体験サンプル。

英語版 (bert_context_demo.py) の日本語版。
同じ単語でも文脈でベクトルが変わることを、日本語の多義語で確認する。

題材: 多義語「アップル」
  グループA: アップル = 果物のりんご
  グループB: アップル = IT企業

実行:
  python bert_context_demo_ja.py
  (初回はモデル約450MBを自動ダウンロード)
  事前に: python -m pip install fugashi unidic-lite
"""

import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"

# 多義語「アップル」: [0][1]=果物 / [2][3]=IT企業
SENTENCES = [
    "アップル を むいて 皮 を 捨てた。",          # 0: 果物
    "甘い アップル を 収穫 した。",               # 1: 果物
    "アップル が 新しい スマホ を 発表 した。",    # 2: 企業
    "アップル の 株価 が 上昇 した。",            # 3: 企業
]
LABELS = ["果物-0", "果物-1", "企業-2", "企業-3"]
TARGET_WORD = "アップル"


def get_word_vector(text, tokenizer, model, target_word):
    """文を BERT でエンコードし、target_word の文脈依存ベクトルを返す。

    日本語は単語が複数サブワードに割れることがあるため、
    target_word を単独でトークン化した列と一致する位置を探し、
    そのサブワード群の平均ベクトルを返す。
    """
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state[0]  # (トークン数, 768)

    ids = inputs["input_ids"][0].tolist()
    # target_word のサブワードID列 (特殊トークンなし)
    target_ids = tokenizer(target_word, add_special_tokens=False)["input_ids"]

    # ids の中から target_ids の並びを探す
    L = len(target_ids)
    for i in range(len(ids) - L + 1):
        if ids[i:i + L] == target_ids:
            return hidden[i:i + L].mean(dim=0)  # サブワード平均
    tokens = tokenizer.convert_ids_to_tokens(ids)
    raise ValueError(f"'{target_word}' が見つからない: {text}\n tokens={tokens}")


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    print(f"モデル読み込み中: {MODEL_NAME} (初回はダウンロードあり)\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    vectors = [get_word_vector(t, tokenizer, model, TARGET_WORD) for t in SENTENCES]

    print("=" * 70)
    print(f"対象単語 '{TARGET_WORD}' の文脈依存ベクトルを各文から抽出")
    print("=" * 70)
    for lab, text in zip(LABELS, SENTENCES):
        print(f"  [{lab}] {text}")

    print("\n" + "=" * 70)
    print(f"'{TARGET_WORD}' ベクトル同士のコサイン類似度 (1.0=同じ意味, 低い=違う意味)")
    print("=" * 70)
    n = len(SENTENCES)
    print("          " + "".join(f"{i:>9}" for i in range(n)))
    for i in range(n):
        row = [f"{LABELS[i]:<10}"]
        for j in range(n):
            row.append(f"{cosine(vectors[i], vectors[j]):>9.3f}")
        print("".join(row))

    same_fruit = cosine(vectors[0], vectors[1])
    same_corp = cosine(vectors[2], vectors[3])
    diff_pairs = [
        cosine(vectors[0], vectors[2]), cosine(vectors[0], vectors[3]),
        cosine(vectors[1], vectors[2]), cosine(vectors[1], vectors[3]),
    ]
    avg_diff = sum(diff_pairs) / len(diff_pairs)

    print("\n" + "=" * 70)
    print("意味グループごとの比較")
    print("=" * 70)
    print(f"  同じ意味 (果物 vs 果物)   : {same_fruit:.3f}")
    print(f"  同じ意味 (企業 vs 企業)   : {same_corp:.3f}")
    print(f"  違う意味 (果物 vs 企業) 平均: {avg_diff:.3f}")

    print("\n" + "=" * 70)
    print("結論")
    print("=" * 70)
    print("  同じ 'アップル' でも、文脈が同じ文どうしは類似度が高く、")
    print("  文脈が違う (果物 vs 企業) 文どうしは類似度が下がる。")
    print("  => 日本語でも BERT は文脈で単語のベクトルを変える = 文脈依存埋め込み。")


if __name__ == "__main__":
    main()
