"""Live voice demo: speak in ANY language -> ASR (whisper) -> Gemma -> Kreol reply -> TTS.

Run on the box:
    /root/bonsai/f5tts/bin/python scripts/live_demo.py --host 0.0.0.0 --port 7860
Then open http://<box-ip>:7860 in a browser, allow microphone access, and talk.
"""
import argparse
import tempfile
import time

import gradio as gr

from gemma_reason import generate
from mms_tts import synthesize
from whisper_asr import SIZES, transcribe

SYSTEM_KREOL_ONLY = (
    "You are a friendly voice assistant for a Mauritian Creole (Kreol Morisien) speaker. "
    "The user may speak to you in ANY language (e.g. English, French, Hindi, Chinese, Spanish). "
    "ALWAYS answer in fluent, natural Kreol Morisien only, no matter what language the user "
    "spoke in. Keep replies short and conversational. If you did not understand, ask a short "
    "clarifying question in Kreol Morisien."
)

RESPONSE_INSTRUCTION = (
    "\n\nUser's spoken message (transcribed):\n{transcript}\n\n"
    "Respond in Kreol Morisien only."
)


def handle(audio_path, model_size):
    t0 = time.time()
    transcript, langs, per, primary = transcribe(audio_path, model_size=model_size)
    asr_t = time.time() - t0

    t1 = time.time()
    reply = generate(SYSTEM_KREOL_ONLY, RESPONSE_INSTRUCTION.format(transcript=transcript))
    reason_t = time.time() - t1

    t2 = time.time()
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    synthesize(reply, out_path)
    tts_t = time.time() - t2

    summary = (f"ASR {asr_t:.1f}s | Gemma {reason_t:.1f}s | TTS {tts_t:.1f}s | "
               f"total {time.time() - t0:.1f}s")
    return transcript, ", ".join(langs), reply, out_path, summary


def build():
    with gr.Blocks(title="Kreol Voice Assistant — any language in, Kreol out") as demo:
        gr.Markdown(
            "# Kreol Voice Assistant\n"
            "Speak in **any language** — you'll get an answer **in Kreol Morisien**.\n"
            "ASR: Whisper (auto-detects language) · Reasoning: Gemma · Voice: mms-tts-mfe."
        )
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(source=["microphone", "upload"], type="filepath",
                                 label="Speak here (any language)")
                model = gr.Dropdown(choices=list(SIZES), value="small", label="Whisper model size")
                btn = gr.Button("Run", variant="primary")
            with gr.Column():
                transcript = gr.Textbox(label="Transcript (as recognized)", lines=3)
                langs = gr.Textbox(label="Detected language(s)")
                reply = gr.Textbox(label="Gemma reply (Kreol Morisien)", lines=4)
                latency = gr.Textbox(label="Latency")
                audio_out = gr.Audio(type="filepath", label="Spoken reply (Kreol)")
        btn.click(handle, [audio, model], [transcript, langs, reply, audio_out, latency])
    return demo


def main():
    ap = argparse.ArgumentParser(description="Live Kreol voice demo (Gradio)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public share link")
    args = ap.parse_args()

    demo = build()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()