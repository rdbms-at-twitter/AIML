"""
Real-time English transcription + Japanese translation
Amazon Transcribe Streaming + Amazon Translate

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
LANGUAGE_CODE = "en-US"
TARGET_LANGUAGE = "ja"

# Amazon Translate client
translate_client = boto3.client("translate", region_name=REGION)


def translate_to_japanese(text: str) -> str:
    try:
        response = translate_client.translate_text(
            Text=text,
            SourceLanguageCode="en",
            TargetLanguageCode=TARGET_LANGUAGE,
        )
        return response["TranslatedText"]
    except Exception as e:
        return f"[Translation error: {e}]"


class TranscribeHandler(TranscriptResultStreamHandler):

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results

        for result in results:
            if not result.is_partial:
                for alt in result.alternatives:
                    transcript = alt.transcript.strip()
                    if transcript:
                        print(f"\n[EN] {transcript}")
                        japanese = translate_to_japanese(transcript)
                        print(f"[JA] {japanese}")
            else:
                for alt in result.alternatives:
                    partial = alt.transcript.strip()
                    if partial:
                        print(f"\r  ... {partial}", end="", flush=True)


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


async def main():
    print("=" * 60)
    print("  Real-time English Transcription + Japanese Translation")
    print("  (Amazon Transcribe Streaming + Amazon Translate)")
    print("=" * 60)
    print()

    client = TranscribeStreamingClient(region=REGION)

    stream = await client.start_stream_transcription(
        language_code=LANGUAGE_CODE,
        media_sample_rate_hz=SAMPLE_RATE,
        media_encoding="pcm",
    )

    handler = TranscribeHandler(stream.output_stream)
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

