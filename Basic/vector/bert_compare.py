# -*- coding: utf-8 -*-
"""
BERT 文脈依存ベクトル 対話比較ツール。

同じ単語を含む2つの文を自由に入力し、その単語のBERTベクトルが
文脈によってどれだけ変わるかをコサイン類似度で比較する。

使い方の流れ:
  1. 起動時に言語(モデル)を選ぶ: 英語 or 日本語
  2. 比較したい「対象単語」を入力
  3. その単語を含む「文1」「文2」を入力
  4. 2文でのその単語のベクトルのコサイン類似度が表示される
  5. 繰り返し比較でき、quit で終了

実行:
  python bert_compare.py

必要ライブラリ:
  python -m pip install transformers torch
  # 日本語モデルを使う場合は追加で:
  python -m pip install fugashi unidic-lite
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModel

MODELS = {
    "en": "bert-base-uncased",
    "ja": "cl-tohoku/bert-base-japanese-v3",
}


def load_model(lang):
    name = MODELS[lang]
    print(f"\nモデル読み込み中: {name} (初回はダウンロードあり) ...")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    print("読み込み完了。\n")
    return tokenizer, model


def get_word_vector(text, tokenizer, model, target_word):
    """文中の target_word の文脈依存ベクトルを返す。
    複数サブワードに割れる場合は平均、複数回出る場合は最初の一致を使う。
    見つからなければ None。
    """
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state[0]

    ids = inputs["input_ids"][0].tolist()
    target_ids = tokenizer(target_word, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return None

    L = len(target_ids)
    for i in range(len(ids) - L + 1):
        if ids[i:i + L] == target_ids:
            return hidden[i:i + L].mean(dim=0)
    return None


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def choose_language():
    while True:
        ans = input("言語を選択してください [en=英語 / ja=日本語] (default en): ").strip().lower()
        if ans == "":
            return "en"
        if ans in MODELS:
            return ans
        print("  en か ja を入力してください。")


def prompt(msg):
    """入力を受け取り、quit/exit なら終了シグナルを返す。"""
    val = input(msg).strip()
    if val.lower() in ("quit", "exit", "q"):
        print("終了します。")
        sys.exit(0)
    return val


def interpret(sim):
    """類似度の目安コメント。"""
    if sim >= 0.85:
        return "非常に高い（ほぼ同じ意味・文脈）"
    if sim >= 0.7:
        return "高い（近い意味・文脈）"
    if sim >= 0.55:
        return "中くらい（やや関連）"
    return "低い（違う意味・文脈の可能性）"


def main():
    print("=" * 70)
    print("BERT 文脈依存ベクトル 対話比較ツール")
    print("同じ単語を含む2文を入力し、その単語のベクトルの近さを比べます。")
    print("（各入力で quit と打つと終了）")
    print("=" * 70)

    lang = choose_language()
    tokenizer, model = load_model(lang)

    example = "bank" if lang == "en" else "アップル"
    print(f"ヒント: まず比較したい単語(例: {example})を入れ、その単語を含む2文を入力します。\n")

    while True:
        print("-" * 70)
        target = prompt("比較したい単語: ")
        if target == "":
            print("  単語が空です。もう一度。")
            continue

        s1 = prompt(f"文1 ('{target}' を含む文): ")
        s2 = prompt(f"文2 ('{target}' を含む文): ")

        v1 = get_word_vector(s1, tokenizer, model, target)
        v2 = get_word_vector(s2, tokenizer, model, target)

        if v1 is None:
            print(f"  [!] 文1 に '{target}' が見つかりませんでした。表記を合わせて再入力してください。")
            continue
        if v2 is None:
            print(f"  [!] 文2 に '{target}' が見つかりませんでした。表記を合わせて再入力してください。")
            continue

        sim = cosine(v1, v2)
        print("\n  --- 結果 ---")
        print(f"  対象単語 : {target}")
        print(f"  文1      : {s1}")
        print(f"  文2      : {s2}")
        print(f"  コサイン類似度 : {sim:.3f}  → {interpret(sim)}")
        print("  （1.0=文脈まで同じ / 低い=文脈が違い意味がずれている）\n")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n終了します。")
