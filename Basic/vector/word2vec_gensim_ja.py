# -*- coding: utf-8 -*-
"""
日本語対応 Word2Vec サンプル (janome 分かち書き + gensim)。

英語版 (word2vec_gensim.py) との唯一の本質的な違いは、
「日本語は単語の区切りがないので、形態素解析で分かち書きする」点。
分かち書き後は英語版とまったく同じ流れで学習・分析できる。

目的:
  - 日本語コーパスで単語ベクトルを学習し、ベクトルの動きを把握する。
  - king-man+woman=queen の日本語版アナロジー (王-男+女=女王) を試す。

実行:
  python word2vec_gensim_ja.py
"""

import numpy as np
from janome.tokenizer import Tokenizer
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. 日本語コーパス
#    英語版と同じく、王族 / 性別 / 動物 の3グループを意味的に配置。
# ---------------------------------------------------------------------------
RAW = [
    "王様 は 強い 男 です",
    "女王 は 賢い 女 です",
    "少年 は 若い 男 です",
    "少女 は 若い 女 です",
    "王子 は 若い 王様 です",
    "王女 は 若い 女王 です",
    "男 は 強い",
    "女 は 賢い",
    "王子 は 少年 でいつか 王様 に なる",
    "王女 は 少女 でいつか 女王 に なる",
    "犬 は 動物 です",
    "猫 は 動物 です",
    "犬 は 大きく 吠える",
    "猫 は 小さく 鳴く",
]

WATCH = ["王様", "女王", "男", "女", "犬", "猫"]

# 意味を持つ品詞だけ残す (助詞・助動詞・記号などを除外)
KEEP_POS = ("名詞", "動詞", "形容詞", "副詞")


# ---------------------------------------------------------------------------
# 2. 分かち書き — ここが日本語対応の核心
# ---------------------------------------------------------------------------
def build_sentences(raw, use_pos_filter=True):
    """
    日本語文を単語リストに分割する。

    今回のコーパスは既にスペース区切りしてあるが、実運用では生の日本語文
    (スペースなし) を janome で分かち書きする。両対応にするため、
    一旦スペースを除去してから janome に渡している。
    """
    tokenizer = Tokenizer()
    sentences = []
    for line in raw:
        text = line.replace(" ", "")  # スペースを除去して「生の日本語文」にする
        words = []
        for token in tokenizer.tokenize(text):
            pos = token.part_of_speech.split(",")[0]  # 品詞の大分類
            if (not use_pos_filter) or pos in KEEP_POS:
                # 表層形ではなく基本形を使うと活用のゆれを吸収できる
                base = token.base_form
                surface = base if base != "*" else token.surface
                words.append(surface)
        sentences.append(words)
    return sentences


# ---------------------------------------------------------------------------
# 3. 学習中のベクトルの動きを追跡するコールバック (英語版と同一)
# ---------------------------------------------------------------------------
class VectorTracker(CallbackAny2Vec):
    def __init__(self, watch):
        self.watch = watch
        self.epoch = 0
        self.prev = {}
        self.history = {w: [] for w in watch}

    def on_epoch_end(self, model):
        self.epoch += 1
        kv = model.wv
        line = [f"[epoch {self.epoch:3d}]"]
        for w in self.watch:
            if w not in kv:
                continue
            cur = kv[w].copy()
            norm = float(np.linalg.norm(cur))
            move = float(np.linalg.norm(cur - self.prev[w])) if w in self.prev else 0.0
            self.prev[w] = cur
            self.history[w].append((self.epoch, norm, move))
            line.append(f"{w}(|v|={norm:.3f},d={move:.3f})")
        if self.epoch % 3 == 0 or self.epoch == 1:
            print("  ".join(line))


# ---------------------------------------------------------------------------
# 4. 学習
# ---------------------------------------------------------------------------
def train(sentences):
    # コーパスが小さいので複製して実質的な学習量を確保 (英語版と同じ工夫)
    # 日本語は1文あたりの語数が少なく学習信号が弱いため英語版より多めに複製
    sentences = sentences * 60
    tracker = VectorTracker(WATCH)
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=2,
        min_count=1,
        sg=1,             # Skip-gram
        negative=5,
        alpha=0.05,
        min_alpha=0.001,
        epochs=30,
        seed=42,
        workers=1,
        callbacks=[tracker],
    )
    return model, tracker


# ---------------------------------------------------------------------------
# 5. 分析
# ---------------------------------------------------------------------------
def analyze(model, tracker):
    kv = model.wv

    print("\n" + "=" * 60)
    print("学習後の類似語 (gensim most_similar)")
    print("=" * 60)
    for w in WATCH:
        if w not in kv:
            print(f"  {w}: 語彙になし")
            continue
        sims = kv.most_similar(w, topn=3)
        sim_str = ", ".join(f"{sw}({s:.2f})" for sw, s in sims)
        print(f"  {w:6s} -> {sim_str}")

    print("\n" + "=" * 60)
    print("アナロジー: 王様 - 男 + 女 = ?")
    print("=" * 60)
    try:
        res = kv.most_similar(positive=["王様", "女"], negative=["男"], topn=3)
        for word, score in res:
            print(f"  => {word} ({score:.3f})")
    except KeyError as e:
        print(f"  (語彙不足: {e})")

    print("\n" + "=" * 60)
    print("ベクトルの動き: 総移動量 (全エポック合計) と最終ノルム")
    print("=" * 60)
    for w in WATCH:
        hist = tracker.history.get(w, [])
        if not hist:
            continue
        total_move = sum(h[2] for h in hist)
        final_norm = hist[-1][1]
        print(f"  {w:6s} 総移動={total_move:.2f}  最終|v|={final_norm:.2f}")


# ---------------------------------------------------------------------------
# 6. 2D 可視化 (PCA) — 日本語ラベルは環境により文字化けするので注記
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

    # 日本語フォントを試行 (見つからなければ英字ラベルにフォールバック)
    jp_font_ok = False
    try:
        from matplotlib import font_manager
        for cand in ["Meiryo", "Yu Gothic", "MS Gothic", "TakaoGothic",
                     "IPAexGothic", "Noto Sans CJK JP", "Hiragino Sans"]:
            found = [f.name for f in font_manager.fontManager.ttflist if f.name == cand]
            if found:
                matplotlib.rcParams["font.family"] = cand
                jp_font_ok = True
                break
    except Exception:
        pass

    kv = model.wv
    words = list(kv.index_to_key)
    X = np.array([kv[w] for w in words])
    coords = PCA(n_components=2, random_state=42).fit_transform(X)

    plt.figure(figsize=(9, 7))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.5)
    for i, (w, (x, y)) in enumerate(zip(words, coords)):
        color = "red" if w in WATCH else "black"
        # 日本語フォントが無ければ番号ラベルにする (凡例を別途 print)
        label = w if jp_font_ok else f"w{i}"
        plt.annotate(label, (x, y), color=color, fontsize=9)
    plt.title("Word2Vec (Japanese) - PCA 2D")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    out = "word2vec_gensim_ja_pca.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    if jp_font_ok:
        print(f"\n埋め込み可視化を保存: {out} (日本語フォント使用)")
    else:
        print(f"\n埋め込み可視化を保存: {out} (日本語フォント未検出のため wN ラベル)")
        print("  ラベル対応:")
        for i, w in enumerate(words):
            print(f"    w{i}={w}", end="  ")
            if (i + 1) % 6 == 0:
                print()
        print()


def main():
    print("日本語 Word2Vec 学習開始 (janome + gensim, Skip-gram)\n")

    sentences = build_sentences(RAW, use_pos_filter=True)
    print("分かち書き結果 (最初の3文):")
    for s in sentences[:3]:
        print("  ", s)
    print()

    model, tracker = train(sentences)
    analyze(model, tracker)
    visualize(model)

    model.save("word2vec_gensim_ja.model")
    print("モデルを保存: word2vec_gensim_ja.model")


if __name__ == "__main__":
    main()
