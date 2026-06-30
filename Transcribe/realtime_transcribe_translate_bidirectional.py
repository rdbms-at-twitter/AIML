"""
Real-time bidirectional transcription + translation
English -> Japanese / Japanese -> English
(Amazon Transcribe Streaming + Amazon Translate)

Usage: python realtime_transcribe_translate.py
Stop: Ctrl+C
"""

import asyncio
import pyaudio
import boto3
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

# --- Config ---
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024 * 2
REGION = "us-east-1"

# Amazon Translate client
translate_client = boto3.client("translate", region_name=REGION)


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    try:
        response = translate_client.translate_text(
            Text=text,
            SourceLanguageCode=source_lang,
            TargetLanguageCode=target_lang,
        )
        return response["TranslatedText"]
    except Exception as e:
        return f"[Translation error: {e}]"


class TranscribeHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream, source_lang, target_lang, source_label, target_label):
        super().__init__(output_stream)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.source_label = source_label
        self.target_label = target_label

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results

        for result in results:
            if not result.is_partial:
                for alt in result.alternatives:
                    transcript = alt.transcript.strip()
                    if transcript:
                        print(f"\n[{self.source_label}] {transcript}")
                        translated = translate_text(
                            transcript, self.source_lang, self.target_lang
                        )
                        print(f"[{self.target_label}] {translated}")
            else:
                for alt in result.alternatives:
                    partial = alt.transcript.strip()
                    if partial:
                        # Truncate long partials to last 60 chars
                        display = partial[-60:] if len(partial) > 60 else partial
                        # Clear line and overwrite
                        print(f"\r\033[K  ... {display}", end="", flush=True)


async def mic_stream(audio_stream):
    pa = pyaudio.PyAudio()

    default_input = pa.get_default_input_device_info()
    print(f"Mic device: {default_input['name']}")
    print(f"Sample rate: {SAMPLE_RATE} Hz / Channels: {CHANNELS}")
    print("-" * 60)
    print("Listening... (Ctrl+C to stop)")
    print("-" * 60)

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            await audio_stream.input_stream.send_audio_event(audio_chunk=data)
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        await audio_stream.input_stream.end_stream()


def select_mode():
    print("=" * 60)
    print("  Real-time Bidirectional Transcription + Translation")
    print("  (Amazon Transcribe Streaming + Amazon Translate)")
    print("=" * 60)
    print()
    print("  [1] English -> Japanese (speak English)")
    print("  [2] Japanese -> English (speak Japanese)")
    print()

    while True:
        choice = input("Select mode (1 or 2): ").strip()
        if choice == "1":
            return "en-US", "en", "ja", "EN", "JA"
        elif choice == "2":
            return "ja-JP", "ja", "en", "JA", "EN"
        else:
            print("  Please enter 1 or 2.")


async def main():
    language_code, source_lang, target_lang, source_label, target_label = select_mode()

    print()
    print(f"  Mode: {source_label} -> {target_label}")
    print()

    client = TranscribeStreamingClient(region=REGION)

    stream = await client.start_stream_transcription(
        language_code=language_code,
        media_sample_rate_hz=SAMPLE_RATE,
        media_encoding="pcm",
    )

    handler = TranscribeHandler(
        stream.output_stream, source_lang, target_lang, source_label, target_label
    )
    mic_task = asyncio.create_task(mic_stream(stream))

    try:
        await handler.handle_events()
    except asyncio.CancelledError:
        pass
    finally:
        mic_task.cancel()
        try:
            await mic_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStopped.")
