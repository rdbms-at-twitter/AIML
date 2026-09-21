import sys
import MeCab

from collections import defaultdict

# MeCab のタガーを1度だけ生成（使い回す）
_tagger = MeCab.Tagger()

def tokenize(sentence):
    """文を形態素解析し、名詞・動詞・形容詞の原形（語彙素）を返す"""
    tokens = []
    node = _tagger.parseToNode(sentence)
    while node:
        features = node.feature.split(',')
        pos = features[0]  # 品詞
        # 内容語（名詞・動詞・形容詞）のみをスコアリング対象にする
        if pos in ('名詞', '動詞', '形容詞'):
            # 原形があれば原形、なければ表層形を使う
            base = features[6] if len(features) > 6 and features[6] != '*' else node.surface
            if base.strip():
                tokens.append(base)
        node = node.next
    return tokens

def simple_summarize(text, num_sentences=3):
    sentences = [s for s in text.split('。') if s.strip()]  # 空文を除外
    word_frequencies = defaultdict(int)

    for sentence in sentences:
        tokens = tokenize(sentence)
        for token in tokens:
            word_frequencies[token] += 1

    sentence_scores = []
    for i, sentence in enumerate(sentences):
        tokens = tokenize(sentence)
        # 文の長さで割って正規化（長い文が有利になりすぎるのを防ぐ）
        score = sum(word_frequencies[token] for token in tokens)
        if tokens:
            score = score / len(tokens)
        sentence_scores.append((i, score))

    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: x[0])

    summary = '。'.join(sentences[i] for i, _ in top_sentences) + '。'
    return summary

# 使用例
text = """
Amazon Aurora (Aurora) は完全マネージド型のリレーショナルデータベースエンジンで、MySQL および PostgreSQL と互換性があります。MySQL と PostgreSQL が、ハイエンドの商用データベースのスピードおよび信頼性と、オープンソースデータベースのシンプルさとコスト効率を併せ持っていることは既にご存じでしょう。既存の MySQL および PostgreSQL データベースで現在使用しているコード、ツール、アプリケーションを Aurora でも使用できます。Aurora は、同等のハードウェアで、標準の PostgreSQL の最大 6 倍のスループットと標準の MySQL の最大 6 倍のスループットを提供します。Aurora には、高性能のストレージサブシステムが含まれています。MySQL と PostgreSQL との互換性のあるデータベースエンジンは、その高速分散ストレージを利用するようにカスタマイズされています。基本ストレージは、必要に応じて自動的に拡張されます。Aurora クラスターボリュームは、最大 256 tebibytes (TiB) のサイズまで増やすことができます。また、Aurora はデータベースのクラスター化とレプリケーションを自動化しスタンダード化します。通常、これらはデータベースの設定と管理に伴う最も困難な作業に属します。
"""

summary = simple_summarize(text, num_sentences=3)
print(summary)

