#!/usr/bin/env python3
"""clip-studio backend: paste a VOD link -> AI finds moments -> vertical clips.

Run locally:  uvicorn app:app --host 0.0.0.0 --port 8000
Data dir:     DATA_DIR env var (default ./data next to this file)
"""
import json
import os
import subprocess
import traceback
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_CLIP_SECONDS = 120  # cap re-rendered clip length (abuse guard)

app = FastAPI(title="clip-studio")

# In-memory registries. Jobs live on disk under DATA_DIR/jobs/{job_id};
# these dicts map ids -> metadata. (Single-user v1: no DB needed.)
JOBS = {}   # job_id -> {status, message, url, max_clips, dir, duration, clips, error}
CLIPS = {}  # clip_id -> {"job": job_id, "clip": clip_dict}


class JobRequest(BaseModel):
    url: str
    max_clips: int = Field(default=8, ge=1, le=20)


class RenderRequest(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    layout: str = "blur"
    captions: bool = True


def _set(job_id, **kw):
    JOBS[job_id].update(kw)


def run_job(job_id):
    """Background pipeline: download -> analyze -> cut. Runs in a worker thread."""
    job = JOBS[job_id]
    jdir = Path(job["dir"])
    try:
        _set(job_id, status="downloading", message="Downloading video (720p)…")
        video, duration = pipeline.download_video(job["url"], jdir)
        job["duration"] = duration
        (jdir / "meta.json").write_text(json.dumps({"duration": duration}))

        _set(job_id, status="analyzing", message="Extracting audio…")
        audio = pipeline.extract_audio(video, jdir)

        _set(job_id, status="analyzing", message="Finding loud moments…")
        peaks = pipeline.find_peaks(audio)
        (jdir / "peaks.json").write_text(json.dumps(peaks))

        _set(job_id, status="analyzing",
             message="Transcribing audio — this is the slow part, hang tight…")
        segments = pipeline.transcribe(audio)
        (jdir / "transcript.json").write_text(json.dumps(segments))

        _set(job_id, status="analyzing", message="Scoring the best moments…")
        moments = pipeline.score_moments(peaks, segments, job["max_clips"])
        if not moments:
            raise RuntimeError("No standout moments found in this video.")

        _set(job_id, status="cutting", message="Cutting vertical clips…")
        clips = []
        for i, m in enumerate(moments):
            _set(job_id, status="cutting",
                 message=f"Cutting clip {i + 1}/{len(moments)}…")
            start, end = pipeline.clip_window(m["sec"], duration)
            ass = jdir / f"clip_{i}.ass"
            pipeline.write_ass(segments, start, end, ass)
            out = jdir / f"clip_{i}.mp4"
            pipeline.render_clip(video, out, start, end,
                                 layout="blur", ass_path=ass)
            clip_id = f"{job_id[:8]}-{i}"
            clip = {
                "id": clip_id,
                "index": i,
                "title": pipeline.moment_title(m),
                "timestamp": pipeline.fmt_ts(m["sec"]),
                "timestamp_sec": m["sec"],
                "start": start,
                "end": end,
                "layout": "blur",
                "captions": True,
                "file": str(out.name),
                "revisions": 0,
                "preview_url": f"/api/clips/{clip_id}/preview",
            }
            clips.append(clip)
            CLIPS[clip_id] = {"job": job_id, "clip": clip}
        _set(job_id, status="done",
             message=f"Done — {len(clips)} clips ready.",
             clips=clips)
    except subprocess.CalledProcessError:
        # yt-dlp / ffmpeg failure — almost always a bad, private, or
        # region-blocked link. Keep it human-readable.
        traceback.print_exc()
        _set(job_id, status="error",
             message=("Couldn't download that video. Check the link — "
                      "private, deleted, or age-restricted videos won't work."))
    except Exception as e:  # noqa: BLE001 - surface any pipeline failure to the UI
        traceback.print_exc()
        _set(job_id, status="error",
             message=f"Something broke: {str(e)[:300]}")


@app.post("/api/jobs")
def create_job(req: JobRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex
    jdir = DATA_DIR / "jobs" / job_id
    jdir.mkdir(parents=True, exist_ok=True)
    JOBS[job_id] = {
        "status": "queued",
        "message": "Queued — starting soon…",
        "url": req.url,
        "max_clips": req.max_clips,
        "dir": str(jdir),
        "duration": None,
        "clips": [],
    }
    bg.add_task(run_job, job_id)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "message": job["message"],
        "duration": job["duration"],
        "clips": job["clips"],
    }


def _clip_or_404(clip_id: str):
    entry = CLIPS.get(clip_id)
    if not entry:
        raise HTTPException(404, "clip not found")
    return entry


@app.get("/api/clips/{clip_id}/preview")
def clip_preview(clip_id: str):
    entry = _clip_or_404(clip_id)
    path = Path(JOBS[entry["job"]]["dir"]) / entry["clip"]["file"]
    if not path.exists():
        raise HTTPException(404, "clip file missing")
    return FileResponse(path, media_type="video/mp4",
                        filename=f"clip-studio-{clip_id}.mp4")


@app.post("/api/clips/{clip_id}/render")
def render_clip_endpoint(clip_id: str, req: RenderRequest):
    entry = _clip_or_404(clip_id)
    job = JOBS[entry["job"]]
    clip = entry["clip"]
    jdir = Path(job["dir"])
    duration = job.get("duration") or pipeline.probe_duration(jdir / "source.mp4")

    if req.layout not in pipeline.LAYOUT_FILTERS:
        raise HTTPException(400, f"unknown layout: {req.layout}")
    if not (0 <= req.start < req.end <= duration):
        raise HTTPException(400, "trim range must satisfy 0 <= start < end <= video length")
    if req.end - req.start > MAX_CLIP_SECONDS:
        raise HTTPException(400, f"clips are capped at {MAX_CLIP_SECONDS}s")

    clip["revisions"] += 1
    out = jdir / f"clip_{clip['index']}_v{clip['revisions']}.mp4"
    ass = None
    if req.captions:
        segments = json.loads((jdir / "transcript.json").read_text())
        ass = jdir / f"clip_{clip['index']}_v{clip['revisions']}.ass"
        pipeline.write_ass(segments, req.start, req.end, ass)
    try:
        pipeline.render_clip(jdir / "source.mp4", out,
                             req.start, req.end,
                             layout=req.layout, ass_path=ass)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"render failed: {str(e)[:200]}")

    clip.update({"start": req.start, "end": req.end,
                 "layout": req.layout, "captions": req.captions,
                 "file": out.name})
    return {"ok": True, "preview_url": clip["preview_url"],
            "file": out.name}


@app.get("/api/layouts")
def layouts():
    return {"layouts": list(pipeline.LAYOUT_FILTERS.keys())}


# ---- frontend (single inlined index.html — no static dir needed) ----
@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")
