import MeCab
import unicodedata
from collections import defaultdict

_tagger = MeCab.Tagger()


# ---------- 表示ユーティリティ（East Asian Width 対応） ----------
def disp_width(s):
    w = 0
    for ch in str(s):
        w += 2 if unicodedata.east_asian_width(ch) in ('F', 'W', 'A') else 1
    return w


def pad(s, width):
    return str(s) + ' ' * (width - disp_width(s))


def print_table(headers, rows):
    cols = list(zip(*([headers] + rows))) if rows else [(h,) for h in headers]
    widths = [max(disp_width(c) for c in col) + 2 for col in cols]

    def sep(l, m, r):
        return l + m.join('─' * w for w in widths) + r

    def line(cells):
        return '│' + '│'.join(' ' + pad(c, w - 2) + ' ' for c, w in zip(cells, widths)) + '│'

    print(sep('┌', '┬', '┐'))
    print(line(headers))
    print(sep('├', '┼', '┤'))
    for r in rows:
        print(line(r))
    print(sep('└', '┴', '┘'))


# ---------- トークン化 ----------
def tokenize(sentence):
    """名詞・動詞・形容詞の原形（内容語）を返す"""
    tokens = []
    node = _tagger.parseToNode(sentence)
    while node:
        f = node.feature.split(',')
        pos = f[0]
        if pos in ('名詞', '動詞', '形容詞'):
            base = f[7] if len(f) > 7 and f[7] != '*' else node.surface
            if base.strip():
                tokens.append(base)
        node = node.next
    return tokens


# ---------- 要約 + 可視化 ----------
def summarize_verbose(text, num_sentences=3):
    sentences = [s.strip() for s in text.split('。') if s.strip()]

    # 1) 各文のトークン化結果
    token_map = {i: tokenize(s) for i, s in enumerate(sentences)}

    # 2) 単語頻度（＝重み）を集計
    word_freq = defaultdict(int)
    for tokens in token_map.values():
        for t in tokens:
            word_freq[t] += 1

    print("\n=== 1. 各文のトークン化結果 ===")
    rows = []
    for i, s in enumerate(sentences):
        preview = s[:20] + ('…' if len(s) > 20 else '')
        rows.append((str(i), preview, str(len(token_map[i])), ' '.join(token_map[i])))
    print_table(('文#', '文（先頭20字）', '語数', 'トークン（内容語の原形）'), rows)

    print("\n=== 2. 単語頻度（重み） 上位20 ===")
    freq_rows = []
    for rank, (word, cnt) in enumerate(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20], 1):
        bar = '█' * cnt
        freq_rows.append((str(rank), word, str(cnt), bar))
    print_table(('順位', '単語', '頻度', 'グラフ'), freq_rows)

    print("\n=== 3. 各文のスコア計算 ===")
    score_rows = []
    scores = []
    for i, s in enumerate(sentences):
        tokens = token_map[i]
        raw = sum(word_freq[t] for t in tokens)
        norm = raw / len(tokens) if tokens else 0.0
        scores.append((i, norm))
        preview = s[:18] + ('…' if len(s) > 18 else '')
        score_rows.append((
            str(i), preview, str(len(tokens)),
            str(raw), f'{norm:.2f}',
            '█' * int(norm * 3)
        ))
    print_table(('文#', '文（先頭18字）', '語数', '頻度合計', '正規化スコア', 'グラフ'), score_rows)

    # 4) 上位文を選定
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:num_sentences]
    top_idx = sorted(i for i, _ in top)

    print(f"\n=== 4. 選定結果（スコア上位{num_sentences}文 → 元順序で復元） ===")
    sel_rows = []
    for i in top_idx:
        score = dict(scores)[i]
        sel_rows.append((str(i), f'{score:.2f}', sentences[i]))
    print_table(('文#', 'スコア', '採用された文'), sel_rows)

    summary = '。'.join(sentences[i] for i in top_idx) + '。'
    print("\n=== 5. 最終要約 ===")
    print(summary)
    return summary


if __name__ == '__main__':
    text = """
Amazon Aurora (Aurora) は完全マネージド型のリレーショナルデータベースエンジンで、MySQL および PostgreSQL と互換性があります。MySQL と PostgreSQL が、ハイエンドの商用データベースのスピードおよび信頼性と、オープンソースデータベースのシンプルさとコスト効率を併せ持っていることは既にご存じでしょう。既存の MySQL および PostgreSQL データベースで現在使用しているコード、ツール、アプリケーションを Aurora でも使用できます。Aurora は、同等のハードウェアで、標準の PostgreSQL の最大 6 倍のスループットと標準の MySQL の最大 6 倍のスループットを提供します。Aurora には、高性能のストレージサブシステムが含まれています。MySQL と PostgreSQL との互換性のあるデータベースエンジンは、その高速分散ストレージを利用するようにカスタマイズされています。基本ストレージは、必要に応じて自動的に拡張されます。Aurora クラスターボリュームは、最大 256 tebibytes (TiB) のサイズまで増やすことができます。また、Aurora はデータベースのクラスター化とレプリケーションを自動化しスタンダード化します。通常、これらはデータベースの設定と管理に伴う最も困難な作業に属します。
"""
    summarize_verbose(text, num_sentences=3)

