import MeCab
import unicodedata


def disp_width(s):
    """East Asian Width を考慮した表示幅を返す（全角=2, 半角=1）"""
    w = 0
    for ch in s:
        w += 2 if unicodedata.east_asian_width(ch) in ('F', 'W', 'A') else 1
    return w


def pad(s, width):
    """表示幅 width になるよう右側にスペースを詰める"""
    return s + ' ' * (width - disp_width(s))


def analyze(text):
    tagger = MeCab.Tagger()
    rows = []
    node = tagger.parseToNode(text)
    while node:
        if node.surface:  # BOS/EOS の空ノードを除外
            f = node.feature.split(',')
            surface = node.surface
            pos = f[0] if len(f) > 0 else ''
            pos_detail = f[1] if len(f) > 1 and f[1] != '*' else ''
            reading = f[6] if len(f) > 6 and f[6] != '*' else ''
            base = f[7] if len(f) > 7 and f[7] != '*' else ''
            rows.append((surface, pos, pos_detail, reading, base))
        node = node.next
    return rows


def print_table(rows):
    headers = ('表層形', '品詞', '品詞細分類', '読み', '原形')
    cols = list(zip(*([headers] + rows))) if rows else [(h,) for h in headers]

    # 各カラムの最大表示幅を算出（左右1スペース分の余白を足す）
    widths = [max(disp_width(str(c)) for c in col) + 2 for col in cols]

    def sep(left, mid, right):
        return left + mid.join('─' * w for w in widths) + right

    def row_line(cells):
        return '│' + '│'.join(' ' + pad(str(c), w - 2) + ' ' for c, w in zip(cells, widths)) + '│'

    print(sep('┌', '┬', '┐'))
    print(row_line(headers))
    print(sep('├', '┼', '┤'))
    for r in rows:
        print(row_line(r))
    print(sep('└', '┴', '┘'))


if __name__ == '__main__':
    text = "mecabの分かち書き"
    rows = analyze(text)
    print_table(rows)

