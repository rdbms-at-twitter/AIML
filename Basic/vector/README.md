# Word2Vec サンプル — ベクトルの動きを把握する

Word2Vec を使って単語ベクトル（word embedding）が学習中にどう「動く」かを体感するためのサンプル集です。
同じコーパスに対して **2つの実装**を用意しており、仕組みの理解と実用の両面から比較できます。

| ファイル | 位置づけ | 依存ライブラリ |
|---|---|---|
| `word2vec_scratch.py` | numpy スクラッチ版（仕組み理解用） | numpy のみ |
| `word2vec_gensim.py` | gensim 本格版（実用・比較用） | gensim / scikit-learn / matplotlib |

---

## 1. Word2Vec とは（前提知識）

Word2Vec は「単語を固定長のベクトル（数値の並び）に変換する」手法です。
**「似た文脈で使われる単語は似たベクトルになる」** という分布仮説に基づいて学習します。

学習が進むと、ベクトル空間上で意味的に近い単語が近くに集まり、次のような演算が成立します。

```
king - man + woman ≒ queen
```

本サンプルでは、この「ベクトルが意味を獲得していく＝空間内を動いていく」様子を観察します。

### 学習アルゴリズム
両サンプルとも **Skip-gram + Negative Sampling** を使用します。

- **Skip-gram**: 中心語から周辺語（文脈）を予測するタスクを解く過程でベクトルを学習する
- **Negative Sampling**: 全語彙で計算する代わりに、正例1つ＋負例数個だけで効率的に学習する近似手法

---

## 2. 共通のコーパス

両サンプルは同じ14文のトイコーパスを使い、結果を比較できるようにしています。

```
the king is a strong man
the queen is a wise woman
the boy is a young man
the girl is a young woman
the prince is a young king
the princess is a young queen
man is strong
woman is wise
prince is a boy will be king
princess is a girl will be queen
dog is an animal
cat is an animal
dog barks loud
cat meows soft
```

意味的に3つのグループ（王族: king/queen/prince/princess、性別: man/woman、動物: dog/cat）が
自然に形成されるよう設計しています。

**追跡対象単語（WATCH）**: `king, queen, man, woman, dog, cat`

---

## 3. `word2vec_scratch.py` — numpy スクラッチ版

### 目的
Word2Vec の**学習の中身を1行ずつ追える**ようにした教育用実装。追加ライブラリ不要（numpy のみ）。

### 構成
| 関数 / クラス | 役割 |
|---|---|
| `tokenize()` | 文を単語ID列に変換し語彙を構築 |
| `build_pairs()` | (中心語, 文脈語) ペアを窓幅 window=2 で生成 |
| `SkipGramNS` | Skip-gram + Negative Sampling 本体 |
| `most_similar()` | コサイン類似度で近い単語を検索 |
| `pca_2d()` | numpy の SVD だけで2次元へ次元削減 |

### 学習の心臓部（ここが「ベクトルの動き」の正体）
```python
score = sigmoid(np.dot(v_c, u_o))   # 中心語ベクトルと文脈語ベクトルの内積 → 確率
g_pos = (score - 1.0)               # 正例の勾配（1に近づけたい）
grad_v = g_pos * u_o
self.W_out[o] -= self.lr * g_pos * v_c   # 文脈語側を更新

# 負例（誤った文脈）は score を 0 に近づける
for n in negs:
    score_n = sigmoid(np.dot(v_c, u_n))
    grad_v += score_n * u_n

self.W_in[c] -= self.lr * grad_v    # ★中心語ベクトルを更新（=動く）
```

- `W_in`: 入力（中心語）側の埋め込み。これが最終的な**単語ベクトル**になる
- `W_out`: 出力（文脈語）側の重み
- 学習率 `lr` は**固定**（0.05）。動きが素直で観察しやすい

### 主なパラメータ
| パラメータ | 値 | 意味 |
|---|---|---|
| `dim` | 10 | ベクトル次元 |
| `neg` | 5 | Negative Sampling 数 |
| `lr` | 0.05 | 学習率（固定） |
| `window` | 2 | 文脈窓の幅 |
| `EPOCHS` | 200 | 学習エポック数 |

### 出力内容
1. 各エポックログ（LOG_EVERY=20ごと）: 各単語の `|vec|`（ノルム）と `移動量Δ`（前回ログからの移動距離）
2. 学習後の類似語（コサイン類似度）
3. アナロジー `king - man + woman = ?`
4. 2D(PCA)空間での開始点→終了点と総移動量
5. matplotlib があれば学習中の軌跡を `word2vec_trajectory.png` に保存

### 実行結果（例）
```
  king     -> queen(1.00), girl(0.87), boy(0.84)
  dog      -> animal(0.77), loud(0.74), cat(0.68)
アナロジー: king - man + woman = ?
  => queen (類似度 0.613)
```

---

## 4. `word2vec_gensim.py` — gensim 本格版

### 目的
実際の Word2Vec 実装（gensim）を使い、**実務に近い形**でベクトルの動きを観察する。

### 構成
| 関数 / クラス | 役割 |
|---|---|
| `VectorTracker` | 各エポック終了時にベクトルのノルム・移動量を記録するコールバック |
| `train()` | `Word2Vec(...)` を呼んで学習 |
| `analyze()` | 類似語・アナロジー・総移動量を表示 |
| `visualize()` | scikit-learn の PCA で全単語を2D可視化して PNG 保存 |

### 学習部分
```python
model = Word2Vec(
    sentences=SENTENCES,
    vector_size=50,   # ベクトル次元
    window=2,         # 文脈窓
    min_count=1,      # 全単語を対象
    sg=1,             # 1=Skip-gram, 0=CBOW
    negative=5,       # Negative Sampling 数
    alpha=0.05,       # 初期学習率
    min_alpha=0.001,  # 最終学習率（自動で線形減衰）
    epochs=30,
    seed=42,
    workers=1,        # 再現性のため単一ワーカー
    callbacks=[tracker],
)
```

学習アルゴリズムはライブラリ内部（C最適化）で実行され、コードには現れません。
代わりに `alpha → min_alpha` の**学習率の自動線形減衰**など実用機能が使えます。

### ベクトルの動きの追跡
`VectorTracker.on_epoch_end()` が各エポックで `model.wv[単語]` を覗き、
前エポックとの差分（移動量）を記録します。

### コーパス複製の注意
```python
SENTENCES = [line.split() for line in RAW] * 30
```
コーパスが14文と極小だと gensim は1エポックあたりの更新が微小でベクトルがほぼ動きません。
学習の動きを観察するため、同じ文を30回複製して実質的な学習量を確保しています（トイ用途の常套手段）。

### 主なパラメータ
| パラメータ | 値 | 意味 |
|---|---|---|
| `vector_size` | 50 | ベクトル次元 |
| `sg` | 1 | Skip-gram（0にすると CBOW） |
| `negative` | 5 | Negative Sampling 数 |
| `alpha` / `min_alpha` | 0.05 / 0.001 | 初期→最終学習率（線形減衰） |
| `epochs` | 30 | 学習エポック数 |

### 出力内容
1. 各エポックログ（3ごと）: 各単語の `|v|`（ノルム）と `d`（移動量）
2. 学習後の類似語（gensim `most_similar`）
3. アナロジー `king - man + woman = ?`
4. 単語ごとの総移動量と最終ノルム
5. 全単語の埋め込みを PCA で2D可視化 → `word2vec_gensim_pca.png`
6. 学習済みモデルを `word2vec_gensim.model` に保存（再利用可）

### 実行結果（例）
```
[epoch   1]  king(|v|=0.091,d=0.000)  ...
[epoch   6]  king(|v|=0.987,d=0.254)  ...   ← 序盤に大きく動く
[epoch  30]  king(|v|=1.333,d=0.003)  ...   ← 収束

  king  -> boy(0.99), prince(0.99), princess(0.98)
  dog   -> animal(0.98), barks(0.97), loud(0.97)
アナロジー: king - man + woman = ?
  => queen (0.975)
```

ノルムが epoch 1→6 で 0.09→1.0 に急成長し、移動量 d が 0.25→0.003 へ減衰していく様子が、
**学習率減衰による「序盤に大きく動き→収束」という実際の Word2Vec らしい挙動**を示しています。

---

## 5. 2つの違い（まとめ）

| 観点 | スクラッチ版 | gensim版 |
|---|---|---|
| 目的 | 仕組みの理解 | 実用 |
| 依存 | numpy のみ | gensim / sklearn / matplotlib |
| アルゴリズム実装 | 自分で全部書く（見える） | ライブラリが隠蔽 |
| 学習率 | 固定（0.05） | 線形減衰（0.05→0.001） |
| ベクトル次元 | 10 | 50 |
| 速度 | 遅い（Pythonループ） | 速い（C最適化） |
| モデル保存 | なし | `.model` に保存可 |
| 動きの見え方 | 更新式そのものを観察 | エポックごとにコールバックで覗く |

### 使い分け
- **学習の更新式を1行ずつ追いたい / 仕組みを学ぶ** → スクラッチ版
- **実データで使う / 大きなコーパスを回す / 保存して再利用** → gensim版

---

## 6. 実行方法

このサンプルは **Windows / Linux / macOS すべてで動作**します。スクリプト自体は OS 依存の
コードを含んでおらず、matplotlib も `matplotlib.use("Agg")` を指定済みなので GUI 環境
（X11 等）がないサーバーでも PNG を保存できます。

### 必要ライブラリのインストール（共通）
```bash
pip install numpy gensim scikit-learn matplotlib
```

### Linux / macOS

パイプもリダイレクトも自由に使えます。UTF-8 がデフォルトのため文字化けや途中終了は起きません。

```bash
python3 word2vec_scratch.py            # スクラッチ版
python3 word2vec_gensim.py             # gensim版

python3 word2vec_gensim.py > out.txt 2>&1   # ログ保存も可
python3 word2vec_gensim.py | tail -40       # パイプも可
```

> ロケールが極端に制限された環境（`LANG=C` や `POSIX` のみ）で日本語 print 時に
> `UnicodeEncodeError` が出る場合のみ、`export PYTHONIOENCODING=utf-8` を付けてください。
> 通常の `LANG=*.UTF-8` 環境では不要です。

### Windows / PowerShell

> **重要**
> gensim版は日本語を出力するため、**パイプやリダイレクト（`|`, `>`, `Out-File`）を使わず**、
> コンソールへ直接出力してください。パイプすると日本語出力の途中でサイレント終了します
> （スクリプトのバグではなく PowerShell のパイプ挙動）。

```powershell
$env:PYTHONIOENCODING="utf-8"
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
cd C:\Users\shinyajp\kiroAgents

python -u word2vec_gensim.py      # gensim版
python word2vec_scratch.py        # スクラッチ版
```

---

## 7. 実験のヒント

パラメータを書き換えてベクトルの動きの変化を観察してみてください。

- `WATCH`（追跡単語）を変える → 別の単語の動きを追える
- `sg=1` → `sg=0` にする → Skip-gram と CBOW の違いを比較
- `epochs` / `alpha` を変える → 収束の速さ・安定性が変わる
- `vector_size`（次元数）を変える → 表現力とノイズのトレードオフ
- コーパスに文を追加する → 新しい意味クラスタが形成される様子を確認

---

## 8. 生成ファイル一覧

| ファイル | 内容 |
|---|---|
| `word2vec_scratch.py` | numpy スクラッチ版スクリプト |
| `word2vec_gensim.py` | gensim 本格版スクリプト |
| `word2vec_gensim.model` | 学習済み gensim モデル（実行後生成） |
| `word2vec_gensim_pca.png` | gensim版の埋め込み2D可視化（実行後生成） |
| `word2vec_trajectory.png` | スクラッチ版の学習軌跡（matplotlib があれば生成） |
