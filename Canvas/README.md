# generate-icon

画像から水彩画風アイコンを生成する Python スクリプト。  
Amazon Bedrock の Nova Canvas モデル (`IMAGE_VARIATION`) を使用しています。

<img width="486" height="489" alt="image" src="https://github.com/user-attachments/assets/3c4f1057-38b5-4b67-b991-dcb59a9f9e5f" />



## Features

- 入力画像を自動で正方形にクロップ
- 小さい画像（320px未満）は自動リサイズ
- Nova Canvas の IMAGE_VARIATION で水彩画風にスタイル変換
- 出力サイズ 1024x1024px（PNG）

## Prerequisites

### AWS

- AWS アカウント
- Amazon Bedrock で **Amazon Nova Canvas** (`amazon.nova-canvas-v1:0`) へのアクセスが有効化されていること
  - Bedrock コンソール → Model access → Amazon Nova Canvas を有効化
- AWS 認証情報が設定済みであること（`~/.aws/credentials` または環境変数）

### Python

- Python 3.9+
- 依存パッケージ:

```bash
pip install boto3 Pillow
```

## Usage

```bash
# 基本（出力先は入力ファイルと同じディレクトリに *_icon.png として保存）
python generate_icon.py <入力画像パス>

# 出力先を指定
python generate_icon.py <入力画像パス> <出力先パス>
```

### Examples

```bash
python generate_icon.py photo.png
python generate_icon.py ./images/profile.jpg ./output/my_icon.png
```

### Output

```
INFO: 入力画像: photo.png
INFO:    元サイズ: 250x334px
INFO:    クロップ: 250x250px (正方形)
INFO:    リサイズ: 320x320px
INFO: アイコン生成中 (水彩画風)...
INFO: ✅ アイコン生成完了: photo_icon.png
INFO:    サイズ: 1024x1024px
INFO:    スタイル: 水彩画風
INFO:    忠実度: 0.75
```

## Configuration

スクリプト上部の定数を編集してカスタマイズできます:

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `ICON_PROMPT` | 水彩画風プロンプト | 生成スタイルを指定するテキスト |
| `NEGATIVE_PROMPT` | ホラー・低品質等を排除 | 避けたい要素を指定 |
| `SIMILARITY_STRENGTH` | `0.75` | 元画像への忠実度（0.2〜1.0） |
| `OUTPUT_SIZE` | `1024` | 出力画像の辺のサイズ（px） |
| `CFG_SCALE` | `8.0` | プロンプトへの追従度 |

### SIMILARITY_STRENGTH の目安

| 値 | 効果 |
|---|---|
| 0.2 - 0.5 | 元画像から大きく変化（別人になる可能性あり） |
| 0.6 - 0.7 | スタイル変換しつつ雰囲気を保持 |
| **0.75** | **バランス型（デフォルト）** |
| 0.8 - 0.9 | 元画像にかなり忠実 |
| 1.0 | ほぼ変化なし |

### プロンプト例（スタイル変更）

```python
# アニメ・イラスト風
ICON_PROMPT = (
    "anime style portrait, colorful, Pixar-like, "
    "friendly expression, clean lines, vibrant colors"
)

# ミニマルベクター
ICON_PROMPT = (
    "minimal vector portrait, flat design, simple shapes, "
    "limited color palette, geometric, modern"
)

# 3Dレンダリング風
ICON_PROMPT = (
    "3D rendered avatar, soft lighting, smooth skin, "
    "Memoji style, friendly, clean background"
)
```

## Supported Input Formats

- PNG
- JPEG / JPG
- 各辺 320〜4096px（320px未満は自動拡大）
- アスペクト比 1:4 〜 4:1

## AWS Region

デフォルトは `us-east-1` です。変更する場合はスクリプト内の `region_name` を編集してください。

## Cost

Nova Canvas の IMAGE_VARIATION は1画像あたり約 $0.04〜$0.08（1024x1024、Standard品質）。  
最新の料金は [Amazon Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) を参照してください。

## License

MIT

