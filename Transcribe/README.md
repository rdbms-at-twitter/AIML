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
