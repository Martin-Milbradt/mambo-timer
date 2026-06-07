#!/usr/bin/env python3
"""Trim leading silence from the mambo audio assets.

The source mp3s shipped with ~450ms of digital silence at the start, which
made the sound play noticeably delayed. This decodes each file, finds the
real onset, keeps a small lead, and re-encodes to mp3 in place.

Usage:
    pip install -r requirements.txt
    python tools/trim_silence.py
"""
import os

import lameenc
import miniaudio
import numpy as np

FILES = ["public/mambo.mp3", "public/full_mambo.mp3"]
KEEP_LEAD_MS = 15       # keep a touch of lead so the attack isn't clipped
ONSET_THRESHOLD = 0.01  # onset = first sample above 1% of peak amplitude
BITRATE = 128


def trim_and_encode(path, keep_lead_ms=KEEP_LEAD_MS, bitrate=BITRATE):
    dec = miniaudio.decode_file(path)
    sr, ch = dec.sample_rate, dec.nchannels
    frames = np.array(dec.samples, dtype=np.int16).reshape(-1, ch)

    amp = np.abs(frames).max(axis=1)
    peak = int(amp.max())
    onset = int(np.argmax(amp > max(1, int(peak * ONSET_THRESHOLD))))
    start = max(0, onset - int(sr * keep_lead_ms / 1000))
    trimmed = frames[start:]

    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sr)
    enc.set_channels(ch)
    enc.set_quality(2)
    mp3 = enc.encode(trimmed.tobytes()) + enc.flush()

    before = os.path.getsize(path)
    with open(path, "wb") as f:
        f.write(mp3)
    print(f"{path}: cut {start / sr * 1000:.0f}ms lead, "
          f"{before} -> {len(mp3)} bytes")


if __name__ == "__main__":
    for f in FILES:
        trim_and_encode(f)
