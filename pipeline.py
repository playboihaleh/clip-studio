#!/usr/bin/env python3
"""clip-studio pipeline: download a VOD, find the best moments, cut vertical clips.

Reuses the approach from ~/workspace/wednesday-myy-dudes/find_moments.py:
  1. yt-dlp download (youtube:player_client=android dodges the bot check)
  2. loudness-spike detection on 1s RMS windows
  3. faster-whisper transcription
  4. keyword matching + merging + scoring

Everything here is importable so app.py can drive it as background jobs.
"""
import json
import os
import re
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

# --- AGENTS.md lesson: strip IPv6 entries from no_proxy (httpx2 bug vs HF hub) ---
# faster-whisper hits the Hugging Face hub at model-load time; a no_proxy entry
# containing "::" breaks it. We scrub those entries process-wide at import.
for _k in list(os.environ):
    if _k.lower() == "no_proxy":
        _kept = [p for p in os.environ[_k].split(",") if ":" not in p.strip()]
        if _kept:
            os.environ[_k] = ",".join(_kept)
        else:
            del os.environ[_k]

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")  # "tiny" on small free hosts
KEYWORDS = re.compile(
    r"(haha|lol|lmao|omg|oh my god|scared|scary|jump scare|scream|no way|what the"
    r"|holy|damn|shit|fuck|wait|clip it|insane|crazy|wtf|let's go|clutch|nice one)",
    re.I,
)

# 1080x1920 vertical layouts. All end in format=yuv420p for H.264.
LAYOUT_FILTERS = {
    # blurred + darkened copy of the video fills the frame, sharp video centered
    "blur": (
        "[0:v]split[a][b];"
        "[a]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,gblur=sigma=40,eq=brightness=-0.45[bg];"
        "[b]scale=1080:-2[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p"
    ),
    # hard crop, centered
    "crop-center": (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,format=yuv420p"
    ),
    # hard crop, keep the left side (useful when facecam is bottom-left)
    "crop-left": (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920:0:0,format=yuv420p"
    ),
    # hard crop, keep the right side
    "crop-right": (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920:in_w-1080:0,format=yuv420p"
    ),
}

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Clip,DejaVu Sans,64,&H00FFFFFF,&H000019FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,0,2,40,40,260,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


# ---------------------------------------------------------------- utilities
def run(cmd, cwd=None):
    subprocess.run(cmd, check=True, cwd=cwd)


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def ytdlp_bin():
    """Locate yt-dlp: PATH first, else python -m yt_dlp."""
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    return ["python3", "-m", "yt_dlp"]


# ---------------------------------------------------------------- download
def download_video(url, outdir):
    """Download best <=720p mp4. Returns (video_path, duration)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run(ytdlp_bin() + [
        "--no-playlist",
        "--extractor-args", "youtube:player_client=android",  # dodge bot check
        "-f", "bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format", "mp4",
        "-o", str(outdir / "source.%(ext)s"),
        url,
    ])
    cands = sorted(outdir.glob("source.*"))
    if not cands:
        raise RuntimeError("yt-dlp produced no file")
    video = cands[0]
    return video, probe_duration(video)


def extract_audio(video_path, outdir):
    """16kHz mono wav for peak detection + whisper."""
    out = Path(outdir) / "audio.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
         "-ar", "16000", "-ac", "1", str(out)])
    return out


# ---------------------------------------------------------------- peaks
def find_peaks(audio_path, min_gap=20, max_peaks=40):
    """1s RMS windows; flag windows louder than the rolling baseline."""
    with wave.open(str(audio_path)) as w:
        n, sr, raw = w.getnframes(), w.getframerate(), w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    win = sr
    rms = np.array([np.sqrt(np.mean(a[i:i + win] ** 2))
                    for i in range(0, len(a), win)])
    db = 20 * np.log10(rms + 1e-9)
    base = np.convolve(db, np.ones(60) / 60, mode="same")
    spike = db - base
    cands = sorted(
        [(t, float(db[t]), float(spike[t])) for t in range(len(db))],
        key=lambda x: -max(x[1] + 18, x[2] - 8))
    peaks = []
    for t, d, s in cands:
        if not (d > -18 or s > 8):
            break
        if all(abs(t - p[0]) >= min_gap for p in peaks):
            peaks.append((t, d, s))
        if len(peaks) >= max_peaks:
            break
    return [{"sec": int(t), "db": round(d, 1), "spike": round(s, 1)}
            for t, d, s in sorted(peaks)]


# ---------------------------------------------------------------- transcribe
def transcribe(audio_path):
    """faster-whisper transcription. Returns [{start, end, text}]."""
    from faster_whisper import WhisperModel  # lazy: heavy import, keeps `import app` light
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), beam_size=5, vad_filter=True)
    return [{"start": round(s.start, 2), "end": round(s.end, 2),
             "text": s.text.strip()} for s in segments]


# ---------------------------------------------------------------- moments
def _text_around(segments, sec, w=18):
    return " ".join(s["text"] for s in segments
                    if s["end"] >= sec - w and s["start"] <= sec + w)


def score_moments(peaks, segments, max_clips):
    """Merge loud peaks + keyword hits, score, return top N sorted by time."""
    hits = [{"sec": p["sec"], "kind": "loud", "spike": p["spike"],
             "text": _text_around(segments, p["sec"])} for p in peaks]
    for s in segments:
        if KEYWORDS.search(s["text"]):
            hits.append({"sec": int(s["start"]), "kind": "keyword",
                         "spike": 0.0, "text": s["text"]})
    hits.sort(key=lambda h: h["sec"])
    merged = []
    for h in hits:
        if merged and h["sec"] - merged[-1]["sec"] < 30:
            # same moment: prefer the loud peak over a bare keyword hit
            if h["kind"] == "loud" and merged[-1]["kind"] == "keyword":
                merged[-1] = h
            continue
        merged.append(h)
    for h in merged:
        kw_bonus = 40 if KEYWORDS.search(h["text"]) else 0
        h["score"] = round((h["spike"] or 0) + kw_bonus, 1)
    top = sorted(merged, key=lambda h: -h["score"])[:max_clips]
    return sorted(top, key=lambda h: h["sec"])


def moment_title(moment):
    text = (moment.get("text") or "").strip()
    if not text:
        return "Loud moment"
    return text[:64] + ("…" if len(text) > 64 else "")


def clip_window(sec, duration, pre=10, post=35):
    start = max(0.0, sec - pre)
    end = min(float(duration), sec + post)
    return round(start, 2), round(end, 2)


def fmt_ts(sec):
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


# ---------------------------------------------------------------- captions
def _ass_time(s):
    s = max(0.0, s)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    return f"{h}:{m:02d}:{s % 60:05.2f}"


def write_ass(segments, clip_start, clip_end, path):
    """Burnable ASS subtitles for segments overlapping [clip_start, clip_end]."""
    lines = [ASS_HEADER]
    for s in segments:
        if s["end"] < clip_start or s["start"] > clip_end:
            continue
        st = _ass_time(s["start"] - clip_start)
        en = _ass_time(s["end"] - clip_start)
        text = s["text"].replace("\n", " ").strip()
        if text:
            lines.append(f"Dialogue: 0,{st},{en},Clip,,0,0,0,,{text}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path


def _esc_filter_path(p):
    # escape for ffmpeg filter parser: backslash, single-quote, colon
    return str(p).replace("\\", "/").replace("'", r"\'").replace(":", r"\:")


# ---------------------------------------------------------------- render
def render_clip(video_path, out_path, start, end, layout="blur", ass_path=None):
    """Cut [start, end) from video_path as a 1080x1920 mp4."""
    if layout not in LAYOUT_FILTERS:
        raise ValueError(f"unknown layout: {layout}")
    if not (0 <= start < end):
        raise ValueError("need 0 <= start < end")
    vf = LAYOUT_FILTERS[layout]
    if ass_path:
        vf += f",subtitles='{_esc_filter_path(ass_path)}'"
    run(["ffmpeg", "-y", "-v", "error",
         "-ss", f"{start:.2f}", "-i", str(video_path),
         "-t", f"{end - start:.2f}",
         "-vf", vf,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
         "-c:a", "aac", "-b:a", "128k",
         str(out_path)])
    return out_path
