# Real-time English Transcription + Japanese Translation

Amazon Transcribe Streaming と Amazon Translate を組み合わせて、マイクからの英語音声をリアルタイムで文字起こしし、同時に日本語へ翻訳するスクリプトです。


- realtime_transcribe_translate.py
<img width="1120" height="326" alt="image" src="https://github.com/user-attachments/assets/a4804109-7f36-49d3-8d31-8589f0bb69e4" />

- realtime_transcribe_translate_bidirectional.py
<img width="744" height="223" alt="image" src="https://github.com/user-attachments/assets/8f877bc4-eedd-47a6-97c6-a0b491de4879" />



## Requirements

### Python Modules

| モジュール | バージョン | 用途 |
|---|---|---|
| pyaudio | 0.2.14+ | マイク音声キャプチャ |
| amazon-transcribe | 0.6.4+ | Transcribe Streaming SDK |
| boto3 | 1.42+ | Amazon Translate 呼び出し |
| asyncio | (標準ライブラリ) | 非同期処理 |

### インストール

```
pip install pyaudio amazon-transcribe boto3
```

### AWS Services

| サービス | 用途 | 料金ページ |
|---|---|---|
| Amazon Transcribe (Streaming) | リアルタイム音声→テキスト変換 | https://aws.amazon.com/transcribe/pricing/ |
| Amazon Translate | 英語→日本語テキスト翻訳 | https://aws.amazon.com/translate/pricing/ |

### AWS IAM Permissions

実行ユーザー/ロールに以下の権限が必要です:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "transcribe:StartStreamTranscription"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "translate:TranslateText"
      ],
      "Resource": "*"
    }
  ]
}
```

### System Requirements

- Python 3.10+
- Windows / macOS / Linux
- マイクデバイス（音声入力用）
- AWS 認証情報が設定済み（ws configure または環境変数 / IAM ロール）

## Usage

```bash
python realtime_transcribe_translate.py
```

停止: Ctrl+C

## Configuration

スクリプト内の以下の変数で設定を変更できます:

| 変数 | デフォルト値 | 説明 |
|---|---|---|
| REGION | us-east-1 | AWS リージョン |
| LANGUAGE_CODE | en-US | 音声入力の言語 |
| TARGET_LANGUAGE | ja | 翻訳先の言語 |
| SAMPLE_RATE | 16000 | サンプルレート (Hz) |


## Note 

Amazon Transcribe : 音声をテキストに自動的に変換してインサイトを得る
https://aws.amazon.com/jp/transcribe/

# Real-time Bidirectional Transcription + Translation + TTS

リアルタイム音声認識 → 翻訳 → 音声読み上げ を行う Python スクリプト。  
英語 ↔ 日本語の双方向通訳をターミナルから実行できます。

## アーキテクチャ

```
マイク入力 → Amazon Transcribe Streaming → Amazon Translate → Amazon Polly → スピーカー出力
                 (音声認識)                   (翻訳)            (音声合成)
```

## 動作モード

| モード | 入力言語 | 出力言語 | Polly ボイス |
|--------|----------|----------|--------------|
| 1 | English | Japanese | Kazuha (Neural) |
| 2 | Japanese | English | Ruth (Neural) |

## 前提条件

### AWS

- AWS アカウントと適切な IAM 権限:
  - `transcribe:StartStreamTranscription`
  - `translate:TranslateText`
  - `polly:SynthesizeSpeech`
- AWS credentials が設定済み（`~/.aws/credentials` または環境変数）
- リージョン: `us-east-1`（スクリプト内で変更可）

### Python

- Python 3.10 以上

### OS

- Windows（`msvcrt` / `winsound` を使用しているため Windows 専用）

## インストール

```bash
pip install pyaudio boto3 amazon-transcribe pydub
```

> **Note**: `pydub` はスクリプト内では直接使用していませんが、依存パッケージとしてインストールしておくと安全です。  
> 音声再生には Windows 標準の `winsound` を使用するため、**ffmpeg や simpleaudio は不要**です。

### PyAudio のインストールに失敗する場合

```bash
pip install pipwin
pipwin install pyaudio
```

## 使い方

```bash
python realtime_transcribe_translate_polly.py
```

1. モードを選択（1: EN→JA / 2: JA→EN）
2. マイクに向かって話す（PC内蔵マイクでもOK）
3. 認識結果と翻訳がリアルタイムで表示される
4. 翻訳結果が Polly によりスピーカーから読み上げられる

### 操作

| キー | 動作 |
|------|------|
| Space | 一時停止 / 再開 |
| Ctrl+C | 終了 |

## 機能

- **リアルタイム音声認識**: Amazon Transcribe Streaming でストリーミング認識
- **部分結果表示**: 認識途中のテキストをインライン表示（確定後に翻訳）
- **自動翻訳**: 確定テキストを Amazon Translate で即座に翻訳
- **音声読み上げ**: 翻訳結果を Amazon Polly (Neural) で読み上げ
- **フィードバック防止**: 読み上げ中はマイク入力を自動ミュート（ループ防止）
- **非同期キュー**: 音声合成・再生は非同期キューで逐次処理（認識処理をブロックしない）

## 注意事項

- スピーカーで読み上げ中は自動的にマイク入力がミュートされます
- PC内蔵マイクでスピーカー出力を拾う場合（外付けマイクなし）、読み上げ後に自動的にミュート解除されるため通常使用に問題ありません
- Teams/Zoom 等の通話音声を通訳したい場合は、「ステレオミキサー」または仮想オーディオデバイス（VB-Audio Virtual Cable 等）をデフォルト入力に設定してください

## コスト

- **Amazon Transcribe Streaming**: $0.024/分
- **Amazon Translate**: $15.00/100万文字
- **Amazon Polly (Neural)**: $16.00/100万文字

※ 2024年時点の us-east-1 料金。最新は [AWS Pricing](https://aws.amazon.com/pricing/) を参照。

