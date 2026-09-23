# -*- coding: utf-8 -*-
"""
gensim を使った Word2Vec サンプル。

目的:
  - 実際の Word2Vec 学習アルゴリズム (Skip-gram / CBOW) を体験する。
  - 学習途中でベクトルが「動く」様子を、コールバックで各エポックごとに追跡する。
  - 学習後のベクトル空間を PCA で 2D 可視化して保存する。

実行:
  python word2vec_gensim.py
"""

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. コーパス (トークン化済みの文リスト)
#    スクラッチ版と同じ内容にして結果を比較しやすくする。
# ---------------------------------------------------------------------------
RAW = [
    "the king is a strong man",
    "the queen is a wise woman",
    "the boy is a young man",
    "the girl is a young woman",
    "the prince is a young king",
    "the princess is a young queen",
    "man is strong",
    "woman is wise",
    "prince is a boy will be king",
    "princess is a girl will be queen",
    "dog is an animal",
    "cat is an animal",
    "dog barks loud",
    "cat meows soft",
]
# コーパスが極小(14文)だと gensim は 1 epoch あたりの更新が微小で
# ベクトルがほぼ動かない。学習の「動き」を観察するため、同じ文を
# 複製して実質的な学習量を増やす (トイ用途の常套手段)。
SENTENCES = [line.split() for line in RAW] * 30

WATCH = ["king", "queen", "man", "woman", "dog", "cat"]


# ---------------------------------------------------------------------------
# 2. 学習の各エポックでベクトルの動きを記録するコールバック
# ---------------------------------------------------------------------------
class VectorTracker(CallbackAny2Vec):
    """各エポック終了時に、注目単語ベクトルのノルムと前回からの移動量を記録する。"""

    def __init__(self, watch):
        self.watch = watch
        self.epoch = 0
        self.prev = {}
        self.history = {w: [] for w in watch}  # (epoch, norm, move)

    def on_epoch_end(self, model):
        self.epoch += 1
        kv = model.wv
        line = [f"[epoch {self.epoch:3d}]"]
        for w in self.watch:
            if w not in kv:
                continue
            cur = kv[w].copy()
            norm = float(np.linalg.norm(cur))
            if w in self.prev:
                move = float(np.linalg.norm(cur - self.prev[w]))
            else:
                move = 0.0
            self.prev[w] = cur
            self.history[w].append((self.epoch, norm, move))
            line.append(f"{w}(|v|={norm:.3f},d={move:.3f})")
        # 3エポックごとにログ
        if self.epoch % 3 == 0 or self.epoch == 1:
            print("  ".join(line))


# ---------------------------------------------------------------------------
# 3. 学習
# ---------------------------------------------------------------------------
def train():
    tracker = VectorTracker(WATCH)
    model = Word2Vec(
        sentences=SENTENCES,
        vector_size=50,   # ベクトル次元
        window=2,         # 文脈窓
        min_count=1,      # トイなので全単語を対象
        sg=1,             # 1=Skip-gram, 0=CBOW
        negative=5,       # negative sampling 数
        alpha=0.05,       # 初期学習率 (動きを見せるため大きめ)
        min_alpha=0.001,  # 最終学習率
        epochs=30,
        seed=42,
        workers=1,        # 再現性のため単一ワーカー
        callbacks=[tracker],
    )
    return model, tracker


# ---------------------------------------------------------------------------
# 4. 分析
# ---------------------------------------------------------------------------
def analyze(model, tracker):
    kv = model.wv

    print("\n" + "=" * 60)
    print("学習後の類似語 (gensim most_similar)")
    print("=" * 60)
    for w in WATCH:
        if w not in kv:
            continue
        sims = kv.most_similar(w, topn=3)
        sim_str = ", ".join(f"{sw}({s:.2f})" for sw, s in sims)
        print(f"  {w:8s} -> {sim_str}")

    print("\n" + "=" * 60)
    print("アナロジー: king - man + woman = ?")
    print("=" * 60)
    try:
        res = kv.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
        for word, score in res:
            print(f"  => {word} ({score:.3f})")
    except KeyError as e:
        print(f"  (語彙不足: {e})")

    print("\n" + "=" * 60)
    print("ベクトルの動き: 各単語の 総移動量 (全エポック合計) と最終ノルム")
    print("=" * 60)
    for w in WATCH:
        hist = tracker.history.get(w, [])
        if not hist:
            continue
        total_move = sum(h[2] for h in hist)
        final_norm = hist[-1][1]
        print(f"  {w:8s} 総移動={total_move:.2f}  最終|v|={final_norm:.2f}")


# ---------------------------------------------------------------------------
# 5. 2D 可視化 (PCA)
# ---------------------------------------------------------------------------
def visualize(model):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        print("\n(matplotlib/sklearn 未導入のため可視化スキップ)")
        return

    kv = model.wv
    words = list(kv.index_to_key)
    X = np.array([kv[w] for w in words])
    coords = PCA(n_components=2, random_state=42).fit_transform(X)

    plt.figure(figsize=(9, 7))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.5)
    for w, (x, y) in zip(words, coords):
        color = "red" if w in WATCH else "black"
        plt.annotate(w, (x, y), color=color, fontsize=9)
    plt.title("Word2Vec (gensim) embedding - PCA 2D")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    out = "word2vec_gensim_pca.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n埋め込み可視化を保存: {out}")


def main():
    print("gensim Word2Vec 学習開始 (Skip-gram)\n")
    model, tracker = train()
    analyze(model, tracker)
    visualize(model)

    # モデル保存 (再利用可能)
    model.save("word2vec_gensim.model")
    print("モデルを保存: word2vec_gensim.model")


if __name__ == "__main__":
    main()
