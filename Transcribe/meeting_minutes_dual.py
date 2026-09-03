"""
リアルタイム議事録ツール（2系統: 自分マイク + 相手の声ループバック）

イヤホン会議で「自分の声（マイク）」と「相手の声（PC出力のループバック）」を
別々に文字起こしし、話者ラベル付きで議事録に残します。

Amazon Transcribe Streaming (+ Amazon Translate)

モード選択:
  1) 英語 → 日本語訳   : 英語を文字起こしして日本語訳を併記
  2) 日本語 → そのまま : 日本語を文字起こししてそのまま記録

Usage: python meeting_minutes_dual.py
Stop:  Ctrl+C

議事録は minutes フォルダに minutes_YYYY-MM-DD_HHMMSS.md として自動保存されます。

必要ライブラリ: PyAudioWPatch, amazon-transcribe, boto3
"""

import asyncio
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
import audioop  # noqa: E402

import pyaudiowpatch as pyaudio  # noqa: E402
import boto3  # noqa: E402
from amazon_transcribe.client import TranscribeStreamingClient  # noqa: E402
from amazon_transcribe.handlers import TranscriptResultStreamHandler  # noqa: E402
from amazon_transcribe.model import TranscriptEvent  # noqa: E402

# --- Config ---
TARGET_RATE = 16000       # Transcribe に送るサンプルレート
TARGET_CHANNELS = 1       # モノラル
CHUNK_FRAMES = 1024 * 2   # 1回に読むフレーム数
REGION = "us-east-1"

# 議事録ファイルの保存先ディレクトリ
MINUTES_DIR = r"C:\Users\shinyajp\kiroAgents\minutes"

# 話者ラベル
SPEAKER_SELF = "🎤 自分"
SPEAKER_OTHER = "🔊 相手"

MODES = {
    "1": {"name": "英語 → 日本語訳", "language_code": "en-US", "translate": True},
    "2": {"name": "日本語 → そのまま", "language_code": "ja-JP", "translate": False},
}

translate_client = boto3.client("translate", region_name=REGION)


def translate_to_japanese(text: str) -> str:
    try:
        response = translate_client.translate_text(
            Text=text, SourceLanguageCode="en", TargetLanguageCode="ja",
        )
        return response["TranslatedText"]
    except Exception as e:
        return f"[Translation error: {e}]"


def select_mode() -> dict:
    print("議事録モードを選択してください:")
    for key, mode in MODES.items():
        print(f"  {key}) {mode['name']}")
    while True:
        choice = input("番号を入力 [1/2]: ").strip()
        if choice in MODES:
            return MODES[choice]
        print("1 または 2 を入力してください。")


class MinutesWriter:
    """議事録を Markdown ファイルに追記していく（話者ラベル付き）"""

    def __init__(self, directory: str, mode: dict):
        os.makedirs(directory, exist_ok=True)
        self.translate = mode["translate"]
        self._lock = asyncio.Lock()
        started = datetime.now()
        filename = f"minutes_{started:%Y-%m-%d_%H%M%S}.md"
        self.path = os.path.join(directory, filename)
        self._file = open(self.path, "a", encoding="utf-8")
        self._write_header(started, mode)

    def _write_header(self, started: datetime, mode: dict):
        self._file.write(f"# 議事録 {started:%Y-%m-%d %H:%M:%S}\n\n")
        self._file.write(f"モード: {mode['name']}（2系統: 自分マイク + 相手ループバック）\n\n")
        if self.translate:
            self._file.write("| 時刻 | 話者 | 発言（日本語訳） | 原文（英語） |\n")
            self._file.write("|------|------|------------------|--------------|\n")
        else:
            self._file.write("| 時刻 | 話者 | 発言 |\n")
            self._file.write("|------|------|------|\n")
        self._file.flush()

    async def add_entry(self, speaker: str, source_text: str, japanese: str = None):
        ts = datetime.now().strftime("%H:%M:%S")

        def cell(s: str) -> str:
            return s.replace("|", "\\|").replace("\n", " ")

        async with self._lock:
            if self.translate:
                self._file.write(
                    f"| {ts} | {speaker} | {cell(japanese)} | {cell(source_text)} |\n"
                )
            else:
                self._file.write(f"| {ts} | {speaker} | {cell(source_text)} |\n")
            self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.write(f"\n\n_終了: {datetime.now():%Y-%m-%d %H:%M:%S}_\n")
            self._file.close()


class TranscribeHandler(TranscriptResultStreamHandler):

    def __init__(self, output_stream, minutes: MinutesWriter, translate: bool, speaker: str):
        super().__init__(output_stream)
        self.minutes = minutes
        self.translate = translate
        self.speaker = speaker

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if not result.is_partial:
                for alt in result.alternatives:
                    transcript = alt.transcript.strip()
                    if not transcript:
                        continue
                    if self.translate:
                        japanese = translate_to_japanese(transcript)
                        print(f"\n[{self.speaker}] {transcript}")
                        print(f"          → {japanese}")
                        await self.minutes.add_entry(self.speaker, transcript, japanese)
                    else:
                        print(f"\n[{self.speaker}] {transcript}")
                        await self.minutes.add_entry(self.speaker, transcript)


def get_devices(p: pyaudio.PyAudio):
    """マイク入力と、既定出力に対応するループバックデバイスを取得"""
    mic = p.get_default_input_device_info()

    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])

    loopback = None
    for lb in p.get_loopback_device_info_generator():
        if default_out["name"] in lb["name"]:
            loopback = lb
            break
    if loopback is None:
        # 見つからなければ最初のループバックを使う
        for lb in p.get_loopback_device_info_generator():
            loopback = lb
            break
    return mic, loopback


async def capture_and_send(p, device_info, audio_stream, label):
    """指定デバイスから読み取り、16kHz/mono/pcm に変換して Transcribe に送信。

    ループバックは無音時にデータを出さないため、Transcribe が 15 秒で
    タイムアウトする。これを防ぐため、音声が無い区間は無音チャンクを
    送り続けてストリームを維持する（キープアライブ）。
    """
    src_rate = int(device_info["defaultSampleRate"])
    src_channels = int(device_info["maxInputChannels"])

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def callback(in_data, frame_count, time_info, status):
        # PyAudio のコールバックスレッドから、asyncio キューへ安全に渡す
        try:
            loop.call_soon_threadsafe(queue.put_nowait, in_data)
        except RuntimeError:
            pass
        return (None, pyaudio.paContinue)

    stream = p.open(
        format=pyaudio.paInt16,
        channels=src_channels,
        rate=src_rate,
        input=True,
        input_device_index=device_info["index"],
        frames_per_buffer=CHUNK_FRAMES,
        stream_callback=callback,
    )
    stream.start_stream()

    rate_state = None  # audioop.ratecv の状態
    # 1 回分の無音チャンク（16kHz / mono / 16bit）。CHUNK_FRAMES 相当の長さ。
    silence = b"\x00\x00" * CHUNK_FRAMES
    # 送信周期（秒）: この間に実音が来なければ無音を送る
    interval = CHUNK_FRAMES / TARGET_RATE

    try:
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=interval)
            except asyncio.TimeoutError:
                # 無音区間 → キープアライブ用の無音を送る
                await audio_stream.input_stream.send_audio_event(audio_chunk=silence)
                continue

            # ステレオ → モノラル
            if src_channels == 2:
                data = audioop.tomono(data, 2, 0.5, 0.5)

            # サンプルレート変換 → 16000Hz
            if src_rate != TARGET_RATE:
                data, rate_state = audioop.ratecv(
                    data, 2, 1, src_rate, TARGET_RATE, rate_state
                )

            await audio_stream.input_stream.send_audio_event(audio_chunk=data)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"\n[{label}] 音声送信エラー: {type(e).__name__}: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        try:
            await audio_stream.input_stream.end_stream()
        except Exception:
            pass


async def run_channel(p, device_info, mode, minutes, speaker):
    """1系統分: Transcribe ストリームを開き、送信タスクとハンドラを走らせる"""
    client = TranscribeStreamingClient(region=REGION)
    stream = await client.start_stream_transcription(
        language_code=mode["language_code"],
        media_sample_rate_hz=TARGET_RATE,
        media_encoding="pcm",
    )
    handler = TranscribeHandler(stream.output_stream, minutes, mode["translate"], speaker)
    send_task = asyncio.create_task(capture_and_send(p, device_info, stream, speaker))
    try:
        await handler.handle_events()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"\n[{speaker}] Transcribe エラー: {type(e).__name__}: {e}")
    finally:
        send_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass


async def main(mode: dict):
    print("=" * 60)
    print("  リアルタイム議事録ツール（自分 + 相手）")
    print(f"  モード: {mode['name']}")
    print("=" * 60)

    p = pyaudio.PyAudio()
    mic, loopback = get_devices(p)

    if loopback is None:
        print("エラー: ループバックデバイスが見つかりません。")
        p.terminate()
        return

    print(f"自分 (マイク)   : {mic['name']}  [{int(mic['defaultSampleRate'])}Hz]")
    print(f"相手 (出力録音) : {loopback['name']}  [{int(loopback['defaultSampleRate'])}Hz]")
    print("-" * 60)

    minutes = MinutesWriter(MINUTES_DIR, mode)
    print(f"議事録ファイル: {minutes.path}")
    print("Listening... (Ctrl+C to stop)")
    print("-" * 60)

    self_task = asyncio.create_task(run_channel(p, mic, mode, minutes, SPEAKER_SELF))
    other_task = asyncio.create_task(run_channel(p, loopback, mode, minutes, SPEAKER_OTHER))

    try:
        await asyncio.gather(self_task, other_task, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        for t in (self_task, other_task):
            t.cancel()
        await asyncio.gather(self_task, other_task, return_exceptions=True)
        minutes.close()
        p.terminate()
        print(f"\n議事録を保存しました: {minutes.path}")


if __name__ == "__main__":
    selected_mode = select_mode()
    try:
        asyncio.run(main(selected_mode))
    except KeyboardInterrupt:
        print("\n\nStopped.")
