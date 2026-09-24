# -*- coding: utf-8 -*-
"""
BERT 文脈依存ベクトル 体験サンプル。

目的:
  - Word2Vec との決定的な違い「同じ単語でも文脈でベクトルが変わる」を数値で体感する。
  - Word2Vec: 'bank' はいつでも同じベクトル (静的埋め込み)
  - BERT     : 'bank' は文脈でベクトルが変わる (文脈依存埋め込み)

やること:
  1. 同じ単語 'bank' を「川岸」の文と「銀行」の文にそれぞれ入れる
  2. BERT で各文をエンコードし、'bank' の位置のベクトルを取り出す
  3. 文脈が似ている文どうし / 違う文どうしで、'bank' ベクトルのコサイン類似度を比較
  4. 「意味が同じ文脈なら似て、違う文脈なら離れる」ことを確認

実行:
  python bert_context_demo.py
  (初回はモデル約440MBを自動ダウンロードする)
"""

import torch
from transformers import BertTokenizer, BertModel

MODEL_NAME = "bert-base-uncased"

# ---------------------------------------------------------------------------
# 対象単語 'bank' を含む文。意味が2グループに分かれる。
#   [0][1] 川岸(river)  /  [2][3] 銀行(money)
# ---------------------------------------------------------------------------
SENTENCES = [
    "I sat by the river bank and watched the water.",   # 0: 川岸
    "The boat is tied near the bank of the river.",      # 1: 川岸
    "I deposited money at the bank yesterday.",          # 2: 銀行
    "The bank approved my loan application.",            # 3: 銀行
]
LABELS = ["川岸(river)-0", "川岸(river)-1", "銀行(money)-2", "銀行(money)-3"]
TARGET_WORD = "bank"


def get_word_vector(text, tokenizer, model, target_word):
    """
    文を BERT でエンコードし、target_word に対応するトークンの
    「文脈を反映したベクトル」を返す。
    """
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # last_hidden_state: (1, トークン数, 768) 各トークンの文脈依存ベクトル
    hidden = outputs.last_hidden_state[0]  # (トークン数, 768)

    # target_word のトークン位置を特定
    token_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    target_id = tokenizer.convert_tokens_to_ids(target_word)

    positions = [i for i, tid in enumerate(token_ids.tolist()) if tid == target_id]
    if not positions:
        raise ValueError(f"'{target_word}' が文に見つからない: {text}\n tokens={tokens}")

    # 複数出たら最初の1つを使う
    return hidden[positions[0]]


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    print(f"モデル読み込み中: {MODEL_NAME} (初回はダウンロードあり)\n")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    model.eval()

    # 各文から 'bank' の文脈依存ベクトルを取得
    vectors = []
    for text in SENTENCES:
        vec = get_word_vector(text, tokenizer, model, TARGET_WORD)
        vectors.append(vec)

    print("=" * 70)
    print(f"対象単語 '{TARGET_WORD}' の文脈依存ベクトルを各文から抽出")
    print("=" * 70)
    for lab, text in zip(LABELS, SENTENCES):
        print(f"  [{lab}] {text}")

    # -------------------------------------------------------------------
    # 全ペアのコサイン類似度を表示
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("'bank' ベクトル同士のコサイン類似度 (1.0=同じ意味, 低い=違う意味)")
    print("=" * 70)
    n = len(SENTENCES)
    # ヘッダ
    header = "            " + "".join(f"{i:>10}" for i in range(n))
    print(header)
    for i in range(n):
        row = [f"{LABELS[i]:<12}"]
        for j in range(n):
            row.append(f"{cosine(vectors[i], vectors[j]):>10.3f}")
        print("".join(row))

    # -------------------------------------------------------------------
    # 同じ意味グループ vs 違う意味グループ の平均類似度
    # -------------------------------------------------------------------
    same_river = cosine(vectors[0], vectors[1])   # 川岸 vs 川岸
    same_money = cosine(vectors[2], vectors[3])    # 銀行 vs 銀行
    diff_pairs = [
        cosine(vectors[0], vectors[2]),
        cosine(vectors[0], vectors[3]),
        cosine(vectors[1], vectors[2]),
        cosine(vectors[1], vectors[3]),
    ]
    avg_diff = sum(diff_pairs) / len(diff_pairs)

    print("\n" + "=" * 70)
    print("意味グループごとの比較")
    print("=" * 70)
    print(f"  同じ意味 (川岸 vs 川岸)   : {same_river:.3f}")
    print(f"  同じ意味 (銀行 vs 銀行)   : {same_money:.3f}")
    print(f"  違う意味 (川岸 vs 銀行) 平均: {avg_diff:.3f}")

    print("\n" + "=" * 70)
    print("結論")
    print("=" * 70)
    print("  同じ 'bank' でも、文脈が同じ文どうしは類似度が高く、")
    print("  文脈が違う (川岸 vs 銀行) 文どうしは類似度が下がる。")
    print("  => Word2Vec なら 'bank' は常に同じ1ベクトル (類似度は必ず1.0) だが、")
    print("     BERT は文脈で 'bank' のベクトルが変わる = 文脈依存埋め込み。")


if __name__ == "__main__":
    main()
