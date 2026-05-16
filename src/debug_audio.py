# Standalone tester for AUDIO_AGENT. Bypasses Streamlit so you can
# iterate on the transcript-stitching / overlap-dedup logic against
# data/conv.wav directly.

import argparse
import os
import time

import librosa
import numpy as np

from audio_agent import (
    AUDIO_AGENT,
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    STEP_SAMPLES,
)


def _default_audio() -> str:
    for ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
        p = f"../data/conv{ext}"
        if os.path.exists(p):
            return p
    return "../data/conv.wav"


def make_state(num_retriev_text: int = 0, num_retriev_img: int = 0):
    return {
        "subject_id": 0,
        "action": "",
        "query": "",
        "allowed_years": None,
        "retrieved_docs": [],
        "retrieved_docs_str": "",
        "num_retriev_text": num_retriev_text,
        "num_retriev_img": num_retriev_img,
        "summary": {},
        "action_history": [],
        "step": 0,
        "question": "",
        "chat_history": [],
        "uploaded_image_path": None,
        "audio_chunk": None,
        "transcript_chunk": "",
        "full_transcript": "",
        "conversation_summary": {},
        "chunk_count": 0,
        "last_summary_at_len": 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=_default_audio())
    ap.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the LLM summarization pass — fastest iteration.",
    )
    ap.add_argument(
        "--realtime",
        action="store_true",
        help="Pace each iteration to real audio speed instead of running flat-out.",
    )
    args = ap.parse_args()

    print(f"Loading {args.audio} …")
    waveform, _ = librosa.load(args.audio, sr=SAMPLE_RATE)
    total = len(waveform)
    print(
        f"  samples={total}, duration={total / SAMPLE_RATE:.1f}s, "
        f"chunk={CHUNK_SAMPLES} samples ({CHUNK_SAMPLES / SAMPLE_RATE:.1f}s), "
        f"step={STEP_SAMPLES} samples ({STEP_SAMPLES / SAMPLE_RATE:.1f}s)"
    )

    state = make_state()
    step_seconds = STEP_SAMPLES / SAMPLE_RATE
    print()
    print("─" * 72)

    for i, start in enumerate(range(0, total, STEP_SAMPLES)):
        tic = time.monotonic()
        chunk = waveform[start : start + CHUNK_SAMPLES]
        state["audio_chunk"] = chunk

        # Waveform stats so we can rule out "audio is silent" issues.
        if len(chunk) > 0:
            peak = float(np.max(np.abs(chunk)))
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        else:
            peak, rms = 0.0, 0.0
        print(f"\n[chunk {i:02d}] samples={len(chunk)} "
              f"({start / SAMPLE_RATE:.1f}s — {(start + len(chunk)) / SAMPLE_RATE:.1f}s)"
              f"  peak={peak:.3f}  rms={rms:.4f}")
        if peak < 0.01:
            print("  ⚠ near-silent chunk (peak < 0.01) — MedASR will likely return empty")

        for event in AUDIO_AGENT.stream(state):
            for node, node_output in event.items():
                state.update(node_output)
                if node == "transcribe":
                    raw = node_output.get("transcript_chunk", "")
                    print(f"  ASR raw  ({len(raw)} chars): {raw!r}")
                    print(f"  stitched : {state['full_transcript']!r}")
                elif node == "summarize" and not args.no_summary:
                    cs = state.get("conversation_summary") or {}
                    nonempty = {k: v for k, v in cs.items() if v}
                    print(f"  summary updated → keys with content: {list(nonempty)}")

        elapsed = time.monotonic() - tic
        print(f"  elapsed: {elapsed:.2f}s")
        if args.realtime:
            remaining = step_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)

    print()
    print("─" * 72)
    print("FINAL TRANSCRIPT:")
    print(state["full_transcript"])
    print()
    print(f"  chunks processed: {state['chunk_count']}")
    print(f"  transcript chars: {len(state['full_transcript'])}")


if __name__ == "__main__":
    main()
