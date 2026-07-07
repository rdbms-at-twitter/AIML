"""
Real-time bidirectional transcription + translation + TTS
English -> Japanese / Japanese -> English
(Amazon Transcribe Streaming + Amazon Translate + Amazon Polly)

Usage: python realtime_transcribe_translate.py
Stop: Ctrl+C
"""

import asyncio
import io
import msvcrt
import tempfile
import os
import wave
import winsound

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

# Polly voice mapping
POLLY_VOICES = {
    "ja": {"VoiceId": "Kazuha", "Engine": "neural"},
    "en": {"VoiceId": "Ruth", "Engine": "neural"},
}

# --- Shared state ---
paused = False
speaking = False  # True while Polly audio is playing (mutes mic input)

# --- AWS Clients ---
translate_client = boto3.client("translate", region_name=REGION)
polly_client = boto3.client("polly", region_name=REGION)

# --- Speech queue ---
speech_queue: asyncio.Queue = asyncio.Queue()


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


def synthesize_and_play(text: str, lang: str):
    """Synthesize speech with Polly and play via winsound (no external deps)."""
    global speaking
    voice_config = POLLY_VOICES.get(lang, POLLY_VOICES["en"])

    tmp_path = None
    try:
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat="pcm",
            VoiceId=voice_config["VoiceId"],
            Engine=voice_config["Engine"],
            SampleRate="16000",
        )
        pcm_data = response["AudioStream"].read()

        # Write PCM to a temporary WAV file for winsound
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(pcm_data)

        # Mute mic during playback to prevent feedback loop
        speaking = True
        # SND_FILENAME: play from file, not SND_ASYNC so it blocks until done
        winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
    except Exception as e:
        print(f"\n[TTS Error] {e}")
    finally:
        speaking = False
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


async def speech_worker():
    """Process speech queue sequentially - one utterance at a time."""
    while True:
        text, lang = await speech_queue.get()
        try:
            await asyncio.to_thread(synthesize_and_play, text, lang)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"\n[Speech worker error] {e}")
        finally:
            speech_queue.task_done()


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

                        # Enqueue for TTS playback
                        await speech_queue.put((translated, self.target_lang))
            else:
                for alt in result.alternatives:
                    partial = alt.transcript.strip()
                    if partial:
                        display = partial[-60:] if len(partial) > 60 else partial
                        print(f"\r\033[K  ... {display}", end="", flush=True)


async def key_listener():
    """Monitor keyboard input and toggle pause on Space key press."""
    global paused
    loop = asyncio.get_event_loop()
    while True:
        key_pressed = await loop.run_in_executor(None, msvcrt.kbhit)
        if key_pressed:
            ch = msvcrt.getch()
            if ch == b" ":
                paused = not paused
                state = "\033[93m\u23f8  PAUSED\033[0m" if paused else "\033[92m\u25b6  RESUMED\033[0m"
                print(f"\r\033[K{state}", flush=True)
        await asyncio.sleep(0.05)


async def mic_stream(audio_stream):
    global paused, speaking
    pa = pyaudio.PyAudio()

    default_input = pa.get_default_input_device_info()
    print(f"Mic device: {default_input['name']}")
    print(f"Sample rate: {SAMPLE_RATE} Hz / Channels: {CHANNELS}")
    print("-" * 60)
    print("Listening... (Space=Pause/Resume, Ctrl+C=Stop)")
    print("TTS enabled: translation results will be spoken aloud")
    print("  (Mic auto-mutes during playback to prevent feedback)")
    print("-" * 60)

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    # Silence chunk to keep Transcribe stream alive during pause/speaking
    silence = b"\x00" * CHUNK_SIZE * 2  # 16-bit PCM silence

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            if not paused and not speaking:
                await audio_stream.input_stream.send_audio_event(audio_chunk=data)
            else:
                # Send silence to keep the stream alive
                await audio_stream.input_stream.send_audio_event(audio_chunk=silence)
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
    print("  Real-time Bidirectional Transcription + Translation + TTS")
    print("  (Amazon Transcribe + Translate + Polly)")
    print("=" * 60)
    print()
    print("  [1] English -> Japanese (speak English, hear Japanese)")
    print("  [2] Japanese -> English (speak Japanese, hear English)")
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
    print(f"  TTS Voice: {POLLY_VOICES[target_lang]['VoiceId']} ({target_lang})")
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

    # Start background tasks
    mic_task = asyncio.create_task(mic_stream(stream))
    key_task = asyncio.create_task(key_listener())
    tts_task = asyncio.create_task(speech_worker())

    try:
        await handler.handle_events()
    except asyncio.CancelledError:
        pass
    finally:
        mic_task.cancel()
        key_task.cancel()
        tts_task.cancel()
        try:
            await mic_task
        except asyncio.CancelledError:
            pass
        try:
            await key_task
        except asyncio.CancelledError:
            pass
        try:
            await tts_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStopped.")
