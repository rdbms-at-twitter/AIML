# -*- coding: utf-8 -*-
"""
Word2Vec (Skip-gram + Negative Sampling) を numpy だけでスクラッチ実装したサンプル。

目的:
  - 学習の過程で単語ベクトルがどう「動く」かを把握する。
  - 追加ライブラリ不要 (numpy のみ) で仕組みを理解する。

出力:
  - 学習中、指定したエポックごとに注目単語ベクトルのノルム・移動距離を表示。
  - 学習後、コサイン類似度で近い単語を表示。
  - 学習中の各エポックのベクトルを2次元(PCA)に落として軌跡を print する。
"""

import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. トイコーパス (意味的にグループ化された文)
# ---------------------------------------------------------------------------
CORPUS = [
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


def tokenize(corpus):
    """文リストを単語IDのリストに変換し、語彙を構築する。"""
    words = []
    for line in corpus:
        words.extend(line.lower().split())
    vocab = sorted(set(words))
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for w, i in word2id.items()}
    tokenized = [[word2id[w] for w in line.lower().split()] for line in corpus]
    return tokenized, word2id, id2word


# ---------------------------------------------------------------------------
# 2. Skip-gram 用の (center, context) ペアを生成
# ---------------------------------------------------------------------------
def build_pairs(tokenized, window=2):
    pairs = []
    for sent in tokenized:
        for i, center in enumerate(sent):
            lo = max(0, i - window)
            hi = min(len(sent), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((center, sent[j]))
    return np.array(pairs, dtype=np.int64)


# ---------------------------------------------------------------------------
# 3. モデル
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class SkipGramNS:
    """Skip-gram + Negative Sampling."""

    def __init__(self, vocab_size, dim=10, neg=5, lr=0.05):
        self.V = vocab_size
        self.dim = dim
        self.neg = neg
        self.lr = lr
        # W_in: 入力(中心語)側の埋め込み。これが「単語ベクトル」として使われる。
        self.W_in = (np.random.rand(vocab_size, dim) - 0.5) / dim
        # W_out: 出力(文脈語)側の重み。
        self.W_out = (np.random.rand(vocab_size, dim) - 0.5) / dim

    def _neg_samples(self, unigram_prob, size):
        return np.random.choice(self.V, size=size, p=unigram_prob)

    def train_epoch(self, pairs, unigram_prob):
        loss = 0.0
        idx = np.random.permutation(len(pairs))
        for k in idx:
            c, o = pairs[k]  # center, context(positive)
            v_c = self.W_in[c]  # (dim,)

            # positive
            u_o = self.W_out[o]
            score = sigmoid(np.dot(v_c, u_o))
            loss += -np.log(score + 1e-10)
            g_pos = (score - 1.0)  # d loss / d score

            grad_v = g_pos * u_o
            self.W_out[o] -= self.lr * g_pos * v_c

            # negative
            negs = self._neg_samples(unigram_prob, self.neg)
            for n in negs:
                if n == o:
                    continue
                u_n = self.W_out[n]
                score_n = sigmoid(np.dot(v_c, u_n))
                loss += -np.log(1.0 - score_n + 1e-10)
                g_neg = score_n
                grad_v += g_neg * u_n
                self.W_out[n] -= self.lr * g_neg * v_c

            # 中心語ベクトルを更新
            self.W_in[c] -= self.lr * grad_v
        return loss / len(pairs)


# ---------------------------------------------------------------------------
# 4. ユーティリティ: 類似度・PCA(2D)
# ---------------------------------------------------------------------------
def most_similar(word, W, word2id, id2word, topn=5):
    if word not in word2id:
        return []
    vec = W[word2id[word]]
    norms = np.linalg.norm(W, axis=1) * np.linalg.norm(vec) + 1e-10
    sims = (W @ vec) / norms
    order = np.argsort(-sims)
    result = []
    for i in order:
        if i == word2id[word]:
            continue
        result.append((id2word[i], float(sims[i])))
        if len(result) >= topn:
            break
    return result


def pca_2d(W):
    """numpy だけで PCA(先頭2成分)。可視化用に2次元へ。"""
    X = W - W.mean(axis=0, keepdims=True)
    # SVD で主成分を求める
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T  # (V, 2)


# ---------------------------------------------------------------------------
# 5. メイン: 学習しながらベクトルの「動き」を追跡
# ---------------------------------------------------------------------------
def main():
    tokenized, word2id, id2word = tokenize(CORPUS)
    vocab_size = len(word2id)
    pairs = build_pairs(tokenized, window=2)

    # negative sampling 用の分布(頻度^0.75 が定番)
    counts = np.bincount(pairs[:, 0], minlength=vocab_size).astype(np.float64)
    unigram = counts ** 0.75
    unigram /= unigram.sum()

    model = SkipGramNS(vocab_size, dim=10, neg=5, lr=0.05)

    # 動きを追跡したい注目単語
    watch = [w for w in ["king", "queen", "man", "woman", "dog", "cat"] if w in word2id]

    print(f"語彙数={vocab_size}, 学習ペア数={len(pairs)}, 次元={model.dim}")
    print(f"追跡単語: {watch}\n")

    prev_vecs = {w: model.W_in[word2id[w]].copy() for w in watch}
    trajectory = {w: [] for w in watch}  # 2D軌跡

    EPOCHS = 200
    LOG_EVERY = 20

    for ep in range(1, EPOCHS + 1):
        loss = model.train_epoch(pairs, unigram)

        # 2D 軌跡を記録
        pts = pca_2d(model.W_in)
        for w in watch:
            trajectory[w].append(pts[word2id[w]].copy())

        if ep % LOG_EVERY == 0 or ep == 1:
            print(f"[epoch {ep:3d}] loss={loss:.4f}")
            for w in watch:
                cur = model.W_in[word2id[w]]
                move = np.linalg.norm(cur - prev_vecs[w])  # 前回ログからの移動量
                norm = np.linalg.norm(cur)
                print(f"    {w:8s} |vec|={norm:.4f}  移動量Delta={move:.4f}")
                prev_vecs[w] = cur.copy()
            print()

    # -------------------------------------------------------------------
    # 学習後: 類似語
    # -------------------------------------------------------------------
    print("=" * 60)
    print("学習後の類似語 (コサイン類似度)")
    print("=" * 60)
    for w in watch:
        sims = most_similar(w, model.W_in, word2id, id2word, topn=3)
        sim_str = ", ".join(f"{sw}({s:.2f})" for sw, s in sims)
        print(f"  {w:8s} -> {sim_str}")

    # -------------------------------------------------------------------
    # 類推 (アナロジー): king - man + woman = ?
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("アナロジー: king - man + woman = ?")
    print("=" * 60)
    if all(w in word2id for w in ["king", "man", "woman"]):
        W = model.W_in
        target = W[word2id["king"]] - W[word2id["man"]] + W[word2id["woman"]]
        norms = np.linalg.norm(W, axis=1) * np.linalg.norm(target) + 1e-10
        sims = (W @ target) / norms
        order = np.argsort(-sims)
        exclude = {word2id[w] for w in ["king", "man", "woman"]}
        for i in order:
            if i in exclude:
                continue
            print(f"  => {id2word[i]} (類似度 {sims[i]:.3f})")
            break

    # -------------------------------------------------------------------
    # 2D 軌跡の start/end を表示 (matplotlib が無くても動くよう print)
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2D(PCA) 空間でのベクトル移動: 開始点 -> 終了点")
    print("=" * 60)
    for w in watch:
        start = trajectory[w][0]
        end = trajectory[w][-1]
        dist = np.linalg.norm(end - start)
        print(f"  {w:8s} ({start[0]:+.2f},{start[1]:+.2f}) -> "
              f"({end[0]:+.2f},{end[1]:+.2f})  総移動={dist:.2f}")

    # matplotlib があれば軌跡を描画して PNG 保存
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        for w in watch:
            arr = np.array(trajectory[w])
            plt.plot(arr[:, 0], arr[:, 1], alpha=0.5)
            plt.scatter(arr[0, 0], arr[0, 1], marker="o")  # start
            plt.scatter(arr[-1, 0], arr[-1, 1], marker="*", s=150)  # end
            plt.annotate(w, (arr[-1, 0], arr[-1, 1]))
        plt.title("Word vector trajectory during training (PCA 2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        out = "word2vec_trajectory.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\n軌跡グラフを保存: {out}")
    except ImportError:
        print("\n(matplotlib 未インストールのため軌跡グラフはスキップ)")


if __name__ == "__main__":
    main()
