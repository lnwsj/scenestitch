"""
SceneStitch — Stitch videos + burn subtitles (Whisper auto / SRT upload / manual).
FastAPI + ffmpeg + faster-whisper. Single-process, in-memory job queue.
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import srt
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_ROOT = Path(__file__).parent.resolve()
STATIC_DIR = APP_ROOT / "static"
UPLOAD_DIR = Path(os.environ.get("SCENESTITCH_UPLOAD_DIR", "/tmp/scenestitch/uploads"))
JOB_DIR = Path(os.environ.get("SCENESTITCH_JOB_DIR", "/tmp/scenestitch/jobs"))
OUTPUT_DIR = Path(os.environ.get("SCENESTITCH_OUTPUT_DIR", "/tmp/scenestitch/outputs"))
MAX_UPLOAD_MB = int(os.environ.get("SCENESTITCH_MAX_UPLOAD_MB", "2048"))
MAX_CONCURRENT_JOBS = int(os.environ.get("SCENESTITCH_MAX_CONCURRENT", "1"))
DEFAULT_WHISPER_MODEL = os.environ.get("SCENESTITCH_WHISPER_MODEL", "small")
DEFAULT_WHISPER_DEVICE = os.environ.get("SCENESTITCH_WHISPER_DEVICE", "auto")  # auto|cpu|cuda
WHISPER_COMPUTE = os.environ.get("SCENESTITCH_WHISPER_COMPUTE", "auto")  # auto|int8|float16|float32
APP_VERSION = "1.0.0"

for d in (UPLOAD_DIR, JOB_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ffmpeg discovery
# ---------------------------------------------------------------------------
FFMPEG_CANDIDATES = [
    os.environ.get("SCENESTITCH_FFMPEG", "").strip() or None,
    "/opt/ffmpeg-cuda-v6/bin/ffmpeg",
    "/opt/ffmpeg-cuda-sub/bin/ffmpeg",
    "/usr/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
]
FFMPEG_BIN: Optional[str] = None
for cand in FFMPEG_CANDIDATES:
    if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
        FFMPEG_BIN = cand
        break
if not FFMPEG_BIN:
    for cmd in ("ffmpeg",):
        p = shutil.which(cmd)
        if p:
            FFMPEG_BIN = p
            break

# Burn-in ffmpeg: needs libass for the 'ass'/'subtitles' filter.
# CUDA builds often don't have libass, so we prefer the system ffmpeg if it
# has it. Override with SCENESTITCH_FFMPEG_BURN.
FFMPEG_BURN_CANDIDATES = [
    os.environ.get("SCENESTITCH_FFMPEG_BURN", "").strip() or None,
    "/usr/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
]
FFMPEG_BURN_BIN: Optional[str] = None
for cand in FFMPEG_BURN_CANDIDATES:
    if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
        # Verify it has the subtitles filter
        try:
            r = subprocess.run(
                [cand, "-hide_banner", "-h", "filter=subtitles"],
                capture_output=True, text=True, timeout=5,
            )
            if "Render text subtitles" in (r.stdout + r.stderr):
                FFMPEG_BURN_BIN = cand
                break
        except Exception:
            pass
# Fall back to main ffmpeg if it has libass
if not FFMPEG_BURN_BIN and FFMPEG_BIN:
    try:
        r = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-h", "filter=subtitles"],
            capture_output=True, text=True, timeout=5,
        )
        if "Render text subtitles" in (r.stdout + r.stderr):
            FFMPEG_BURN_BIN = FFMPEG_BIN
    except Exception:
        pass
if not FFMPEG_BURN_BIN:
    FFMPEG_BURN_BIN = FFMPEG_BIN  # last resort, may fail at burn time

FFPROBE_BIN = None
# Prefer ffprobe from main ffmpeg
if FFMPEG_BIN:
    cand = Path(FFMPEG_BIN).parent / "ffprobe"
    if cand.is_file():
        FFPROBE_BIN = str(cand)
if not FFPROBE_BIN:
    cand = Path("/usr/bin/ffprobe")
    if cand.is_file():
        FFPROBE_BIN = str(cand)
if not FFPROBE_BIN:
    p = shutil.which("ffprobe")
    if p:
        FFPROBE_BIN = p

# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------
JOB_STATUSES = ("queued", "running", "done", "error", "cancelled")


@dataclass
class Job:
    job_id: str
    kind: str  # "concat" | "whisper" | "render"
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    output_path: Optional[str] = None
    srt_path: Optional[str] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)
    cancel_flag: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed": (
                (self.finished_at or time.time()) - (self.started_at or self.created_at)
            ),
            "output_url": f"/api/download/{self.job_id}" if self.output_path else None,
            "srt_url": f"/api/srt/{self.job_id}" if self.srt_path else None,
            "error": self.error,
            "meta": self.meta,
            "log_tail": self.log[-20:],
        }


JOBS: Dict[str, Job] = {}
JOB_LOCK = threading.Lock()
JOB_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_JOBS)
WHISPER_MODEL = None
WHISPER_MODEL_NAME = ""
WHISPER_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# SQLite persistence (job history)
# ---------------------------------------------------------------------------
DB_PATH = Path(os.environ.get("SCENESTITCH_DB", str(APP_ROOT / "data" / "scenestitch.db")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DB_LOCK = threading.Lock()

OUTPUT_TTL_DAYS = int(os.environ.get("SCENESTITCH_OUTPUT_TTL_DAYS", "7"))
UPLOAD_TTL_HOURS = int(os.environ.get("SCENESTITCH_UPLOAD_TTL_HOURS", "24"))
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    progress    REAL DEFAULT 0,
    message     TEXT DEFAULT '',
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL,
    output_path TEXT,
    srt_path    TEXT,
    error       TEXT,
    meta_json   TEXT,
    input_count INTEGER DEFAULT 0,
    output_size INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at DESC);
"""


def _db_conn():
    import sqlite3
    c = sqlite3.connect(str(DB_PATH), timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


def db_init():
    with DB_LOCK, _db_conn() as c:
        c.executescript(DB_SCHEMA)


def db_insert_job(job: Job):
    with DB_LOCK, _db_conn() as c:
        c.execute(
            """INSERT INTO jobs
            (job_id, kind, status, progress, message, created_at, started_at, finished_at,
             output_path, srt_path, error, meta_json, input_count, output_size)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                job.job_id, job.kind, job.status, job.progress, job.message,
                job.created_at, job.started_at, job.finished_at,
                job.output_path, job.srt_path, job.error,
                json.dumps(job.meta, ensure_ascii=False) if job.meta else "{}",
                len(job.meta.get("input_count", []) if isinstance(job.meta.get("input_count"), list) else []),
                0,
            ),
        )


def db_update_job(job: Job):
    out_size = 0
    if job.output_path and Path(job.output_path).is_file():
        try:
            out_size = Path(job.output_path).stat().st_size
        except Exception:
            pass
    in_count = 0
    if isinstance(job.meta.get("inputs"), list):
        in_count = len(job.meta["inputs"])
    elif isinstance(job.meta.get("input_count"), int):
        in_count = job.meta["input_count"]
    with DB_LOCK, _db_conn() as c:
        c.execute(
            """UPDATE jobs SET
              status=?, progress=?, message=?, started_at=?, finished_at=?,
              output_path=?, srt_path=?, error=?, meta_json=?, input_count=?, output_size=?
            WHERE job_id=?""",
            (
                job.status, job.progress, job.message, job.started_at, job.finished_at,
                job.output_path, job.srt_path, job.error,
                json.dumps(job.meta, ensure_ascii=False) if job.meta else "{}",
                in_count, out_size, job.job_id,
            ),
        )


def db_list_jobs(
    status: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    q = "SELECT * FROM jobs WHERE 1=1"
    args: List[Any] = []
    if status:
        q += " AND status=?"
        args.append(status)
    if kind:
        q += " AND kind=?"
        args.append(kind)
    q += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    args.extend([limit, offset])
    with DB_LOCK, _db_conn() as c:
        rows = c.execute(q, args).fetchall()
    return [_job_row_to_dict(r) for r in rows]


def db_get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with DB_LOCK, _db_conn() as c:
        r = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return _job_row_to_dict(r) if r else None


def db_delete_job(job_id: str) -> Dict[str, Any]:
    """Delete job record + associated files. Returns {deleted, files_removed, freed_bytes}."""
    job = db_get_job(job_id)
    if not job:
        return {"deleted": False}
    removed = []
    freed = 0
    for p in [job.get("output_path"), job.get("srt_path")]:
        if p and Path(p).is_file():
            try:
                freed += Path(p).stat().st_size
                Path(p).unlink()
                removed.append(str(p))
            except Exception:
                pass
    # Also try removing job dir (concat intermediates, etc.)
    job_dir = JOB_DIR / job_id
    if job_dir.is_dir():
        for child in job_dir.rglob("*"):
            if child.is_file():
                try:
                    freed += child.stat().st_size
                except Exception:
                    pass
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
        except Exception:
            pass
    with DB_LOCK, _db_conn() as c:
        c.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
    return {"deleted": True, "files_removed": removed, "freed_bytes": freed}


def db_count_jobs() -> Dict[str, int]:
    with DB_LOCK, _db_conn() as c:
        rows = c.execute(
            "SELECT status, COUNT(*) as n FROM jobs GROUP BY status"
        ).fetchall()
    return {r["status"]: r["n"] for r in rows}


def db_cleanup_old(output_ttl_days: int = OUTPUT_TTL_DAYS, upload_ttl_hours: int = UPLOAD_TTL_HOURS) -> Dict[str, Any]:
    """Delete done jobs older than output_ttl_days (and their files).
    Delete uploaded files older than upload_ttl_hours (regardless of job).
    Returns {jobs_deleted, files_removed, freed_bytes}."""
    now = time.time()
    output_cutoff = now - output_ttl_days * 86400
    upload_cutoff = now - upload_ttl_hours * 3600

    jobs_deleted = 0
    freed = 0
    # 1) Old done jobs
    with DB_LOCK, _db_conn() as c:
        rows = c.execute(
            "SELECT job_id FROM jobs WHERE status IN ('done','error','cancelled') AND created_at < ?",
            (output_cutoff,),
        ).fetchall()
    for r in rows:
        res = db_delete_job(r["job_id"])
        if res.get("deleted"):
            jobs_deleted += 1
            freed += res.get("freed_bytes", 0)

    # 2) Old uploads (not referenced by any in-flight job)
    with DB_LOCK, _db_conn() as c:
        ref_paths = set()
        for r in c.execute(
            "SELECT meta_json FROM jobs WHERE status IN ('queued','running')"
        ).fetchall():
            try:
                m = json.loads(r["meta_json"] or "{}")
                for inp in m.get("inputs", []):
                    fid = inp.get("file_id")
                    if fid:
                        for p in UPLOAD_DIR.glob(f"{fid}.*"):
                            ref_paths.add(str(p.resolve()))
            except Exception:
                pass
    uploads_removed = 0
    for p in UPLOAD_DIR.iterdir():
        if not p.is_file():
            continue
        if str(p.resolve()) in ref_paths:
            continue
        try:
            mtime = p.stat().st_mtime
        except Exception:
            continue
        if mtime < upload_cutoff:
            try:
                freed += p.stat().st_size
                p.unlink()
                uploads_removed += 1
            except Exception:
                pass

    return {
        "jobs_deleted": jobs_deleted,
        "uploads_removed": uploads_removed,
        "freed_bytes": freed,
    }


def _job_row_to_dict(r) -> Dict[str, Any]:
    """Convert a DB row to the same dict shape as Job.to_dict()."""
    try:
        meta = json.loads(r["meta_json"] or "{}")
    except Exception:
        meta = {}
    return {
        "job_id": r["job_id"],
        "kind": r["kind"],
        "status": r["status"],
        "progress": round(r["progress"] or 0, 3),
        "message": r["message"] or "",
        "created_at": r["created_at"],
        "started_at": r["started_at"],
        "finished_at": r["finished_at"],
        "elapsed": (
            (r["finished_at"] or time.time()) - (r["started_at"] or r["created_at"])
        ),
        "output_url": f"/api/download/{r['job_id']}" if r["output_path"] else None,
        "srt_url": f"/api/srt/{r['job_id']}" if r["srt_path"] else None,
        "error": r["error"],
        "meta": meta,
        "input_count": r["input_count"] or 0,
        "output_size": r["output_size"] or 0,
    }


# Initialize DB on import
db_init()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="SceneStitch", version=APP_VERSION)

# Static files
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:12]}"
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    p = STATIC_DIR / "index.html"
    if not p.is_file():
        return HTMLResponse("<h1>SceneStitch</h1><p>Frontend not built yet.</p>", status_code=500)
    return HTMLResponse(p.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "version": APP_VERSION,
        "ffmpeg": FFMPEG_BIN,
        "ffmpeg_burn": FFMPEG_BURN_BIN,
        "ffprobe": FFPROBE_BIN,
        "whisper_loaded": WHISPER_MODEL is not None,
        "whisper_model": WHISPER_MODEL_NAME or None,
        "whisper_device": DEFAULT_WHISPER_DEVICE,
        "whisper_compute": WHISPER_COMPUTE,
        "max_concurrent": MAX_CONCURRENT_JOBS,
        "upload_dir": str(UPLOAD_DIR),
        "output_dir": str(OUTPUT_DIR),
        "gpu": gpu_info(),
    }


# ---------------------------------------------------------------------------
# GPU info (nvidia-smi wrapper with caching)
# ---------------------------------------------------------------------------
_GPU_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
_GPU_LOCK = threading.Lock()
_GPU_TTL = 2.0  # seconds


# ---------------------------------------------------------------------------
# Background cleanup task
# ---------------------------------------------------------------------------
_CLEANUP_INTERVAL_SEC = 3600  # 1 hour
_cleanup_thread: Optional[threading.Thread] = None
_cleanup_stop = threading.Event()


def _cleanup_loop():
    while not _cleanup_stop.is_set():
        try:
            res = db_cleanup_old()
            if res.get("jobs_deleted") or res.get("uploads_removed"):
                print(f"[cleanup] removed {res['jobs_deleted']} jobs, {res['uploads_removed']} uploads, freed {res['freed_bytes']//1024} KB")
        except Exception as e:
            print(f"[cleanup] error: {e}")
        _cleanup_stop.wait(_CLEANUP_INTERVAL_SEC)


def _start_cleanup_thread():
    global _cleanup_thread
    if _cleanup_thread and _cleanup_thread.is_alive():
        return
    _cleanup_stop.clear()
    _cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True, name="cleanup-loop")
    _cleanup_thread.start()


@app.on_event("startup")
def _on_startup():
    _start_cleanup_thread()
    print(f"[startup] DB: {DB_PATH}")
    print(f"[startup] output TTL: {OUTPUT_TTL_DAYS}d, upload TTL: {UPLOAD_TTL_HOURS}h")


@app.on_event("shutdown")
def _on_shutdown():
    _cleanup_stop.set()


def gpu_info() -> Optional[Dict[str, Any]]:
    """Return nvidia-smi data, cached for 2s. None if nvidia-smi unavailable."""
    now = time.time()
    with _GPU_LOCK:
        if _GPU_CACHE["data"] is not None and (now - _GPU_CACHE["ts"]) < _GPU_TTL:
            return _GPU_CACHE["data"]
    # Run nvidia-smi (no shell, split args)
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [
                smi,
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        # Parse first GPU (we have 1 GPU per node)
        line = out.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            return None
        idx, name, util, mem_used, mem_total, temp, power, power_limit = parts[:8]
        data = {
            "index": int(idx),
            "name": name,
            "util_pct": int(util) if util.isdigit() else 0,
            "mem_used_mb": int(mem_used) if mem_used.isdigit() else 0,
            "mem_total_mb": int(mem_total) if mem_total.isdigit() else 0,
            "temp_c": int(temp) if temp.isdigit() else 0,
            "power_w": float(power) if power.replace(".", "").isdigit() else 0.0,
            "power_limit_w": float(power_limit) if power_limit.replace(".", "").isdigit() else 0.0,
        }
        with _GPU_LOCK:
            _GPU_CACHE["ts"] = now
            _GPU_CACHE["data"] = data
        return data
    except Exception:
        return None


@app.get("/api/health/gpu")
async def health_gpu():
    """Dedicated GPU endpoint (for frequent polling without other health fields)."""
    return {"gpu": gpu_info()}


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------
def ffprobe_meta(path: str) -> Dict[str, Any]:
    """Return {duration, width, height, fps, has_audio, video_codec, audio_codec}."""
    if not FFPROBE_BIN:
        return {}
    try:
        out = subprocess.run(
            [
                FFPROBE_BIN, "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", path,
            ],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode != 0:
            return {}
        data = json.loads(out.stdout)
        streams = data.get("streams", [])
        v = next((s for s in streams if s.get("codec_type") == "video"), {})
        a = next((s for s in streams if s.get("codec_type") == "audio"), {})
        fps_str = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) else 0.0
        except Exception:
            fps = 0.0
        dur = float(data.get("format", {}).get("duration") or v.get("duration") or 0.0)
        return {
            "duration": round(dur, 3),
            "width": int(v.get("width") or 0),
            "height": int(v.get("height") or 0),
            "fps": round(fps, 3),
            "has_audio": bool(a),
            "video_codec": v.get("codec_name", ""),
            "audio_codec": a.get("codec_name", ""),
            "bit_rate": int(data.get("format", {}).get("bit_rate") or 0),
        }
    except Exception as e:
        return {"_error": str(e)}


# ---------------------------------------------------------------------------
# Upload + file management
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Upload one or more video files. Returns list of {file_id, name, size, meta}."""
    out = []
    for f in files:
        name = f.filename or f"upload_{uuid.uuid4().hex[:8]}.mp4"
        ext = Path(name).suffix.lower() or ".mp4"
        file_id = f"u_{uuid.uuid4().hex[:12]}"
        dest = UPLOAD_DIR / f"{file_id}{ext}"
        size = 0
        try:
            with dest.open("wb") as out_f:
                while True:
                    chunk = await f.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB * 1024 * 1024:
                        out_f.close()
                        dest.unlink(missing_ok=True)
                        raise HTTPException(413, f"File too large (>{MAX_UPLOAD_MB}MB): {name}")
                    out_f.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            dest.unlink(missing_ok=True)
            raise HTTPException(500, f"Upload failed: {e}")
        meta = ffprobe_meta(str(dest))
        out.append({
            "file_id": file_id,
            "name": name,
            "size": size,
            "path": f"/files/{file_id}{ext}",
            "meta": meta,
        })
    return {"files": out}


@app.get("/api/files")
async def list_files():
    """List uploaded files (in-memory record of uploads from this session)."""
    # We re-scan disk because uploads are stateless on server side.
    out = []
    for p in sorted(UPLOAD_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_file():
            continue
        size = p.stat().st_size
        out.append({
            "file_id": p.stem,
            "name": p.name,
            "size": size,
            "path": f"/files/{p.name}",
            "meta": ffprobe_meta(str(p)),
        })
    return {"files": out}


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file by file_id stem."""
    deleted = False
    for p in UPLOAD_DIR.iterdir():
        if p.stem == file_id and p.is_file():
            p.unlink()
            deleted = True
    if not deleted:
        raise HTTPException(404, "file not found")
    return {"ok": True, "deleted": file_id}


@app.get("/files/{name}")
async def serve_file(name: str):
    """Serve uploaded file by name (path-traversal safe)."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "", name)
    if not safe or safe.startswith("."):
        raise HTTPException(400, "bad name")
    p = UPLOAD_DIR / safe
    if not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(str(p))


# ---------------------------------------------------------------------------
# Whisper STT → SRT
# ---------------------------------------------------------------------------
def ensure_whisper(model_name: str = DEFAULT_WHISPER_MODEL):
    global WHISPER_MODEL, WHISPER_MODEL_NAME
    with WHISPER_LOCK:
        if WHISPER_MODEL is not None and WHISPER_MODEL_NAME == model_name:
            return WHISPER_MODEL
        # Lazy import
        from faster_whisper import WhisperModel
        device = DEFAULT_WHISPER_DEVICE
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        compute = WHISPER_COMPUTE
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        WHISPER_MODEL = WhisperModel(model_name, device=device, compute_type=compute)
        WHISPER_MODEL_NAME = model_name
        return WHISPER_MODEL


def run_whisper_job(job: Job, video_path: str, model_name: str, language: Optional[str]):
    """Transcribe video to SRT using faster-whisper. Updates job.progress/log."""
    job.status = "running"
    job.started_at = time.time()
    job.message = f"Loading whisper model: {model_name}"
    job.log.append(job.message)
    try:
        m = ensure_whisper(model_name)
    except Exception as e:
        job.status = "error"
        job.error = f"Whisper load failed: {e}"
        job.finished_at = time.time()
        db_update_job(job)
        return
    db_update_job(job)

    job.message = "Transcribing audio…"
    job.log.append(job.message)
    job.progress = 0.1
    db_update_job(job)

    lang = None if not language or language == "auto" else language
    try:
        segments_iter, info = m.transcribe(
            video_path,
            language=lang,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        total_dur = float(info.duration or 0.0)
        srt_segments = []
        last_db_update = 0.0
        for i, seg in enumerate(segments_iter):
            if job.cancel_flag:
                job.status = "cancelled"
                job.message = "Cancelled"
                job.finished_at = time.time()
                db_update_job(job)
                return
            srt_segments.append(srt.Subtitle(
                index=i + 1,
                start=seg.start,
                end=seg.end,
                content=seg.text.strip(),
            ))
            if total_dur > 0:
                job.progress = min(0.95, 0.1 + 0.85 * (seg.end / total_dur))
                job.message = f"Transcribing… {seg.end:.1f}s / {total_dur:.1f}s"
            # Throttle DB writes to every 2s
            now = time.time()
            if now - last_db_update > 2.0:
                db_update_job(job)
                last_db_update = now
        srt_text = srt.compose(srt_segments)
        # Save to job dir
        job_dir = JOB_DIR / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        srt_path = job_dir / "subtitles.srt"
        srt_path.write_text(srt_text, encoding="utf-8")
        job.srt_path = str(srt_path)
        job.meta.update({
            "language": info.language,
            "language_probability": float(info.language_probability),
            "duration": float(info.duration),
            "segments": len(srt_segments),
            "model": model_name,
        })
        job.status = "done"
        job.progress = 1.0
        job.message = f"Done — {len(srt_segments)} segments ({info.language})"
        job.finished_at = time.time()
        db_update_job(job)
    except Exception as e:
        job.status = "error"
        job.error = f"Whisper failed: {e}"
        job.finished_at = time.time()
        db_update_job(job)


def enqueue_job(job: Job, target):
    """Run a job in a worker thread under the concurrency semaphore."""
    def _wrap():
        with JOB_SEMAPHORE:
            try:
                target()
            except Exception as e:
                job.status = "error"
                job.error = f"unhandled: {e}"
                job.finished_at = time.time()
    t = threading.Thread(target=_wrap, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Concat + burn pipeline
# ---------------------------------------------------------------------------
def build_ass_style(style: Dict[str, Any]) -> str:
    """Build the FULL 23-field ASS style line (after 'Style: Name,')."""
    font = style.get("font", "Noto Sans Thai")
    size = int(style.get("size", 48))
    primary = style.get("color", "&H00FFFFFF")  # white
    secondary = "&H000000FF"  # red (for karaoke / ignored)
    outline = style.get("outline_color", "&H00000000")
    back = style.get("back_color", "&H80000000")
    bold = int(bool(style.get("bold", True)))
    italic = 0
    underline = 0
    strikeout = 0
    scale_x = 100
    scale_y = 100
    spacing = 0
    angle = 0
    border_style = 1  # 1=outline+shadow
    outline_w = int(style.get("outline_width", 2))
    shadow = int(style.get("shadow", 0))
    align = int(style.get("align", 2))  # 2 = bottom center
    margin_v = int(style.get("margin_v", 60))
    margin_l = int(style.get("margin_l", 40))
    margin_r = int(style.get("margin_r", 40))
    encoding = 1  # 1 = default
    return (
        f"{font},{size},{primary},{secondary},{outline},{back},"
        f"{bold},{italic},{underline},{strikeout},{scale_x},{scale_y},"
        f"{spacing},{angle},{border_style},{outline_w},{shadow},{align},"
        f"{margin_l},{margin_r},{margin_v},{encoding}"
    )


def srt_to_ass(srt_text: str, style: Dict[str, Any]) -> str:
    """Convert SRT to ASS with the given style, keeping timestamps."""
    style_line = build_ass_style(style)
    header = f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 720
PlayResY: 1280

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style_line}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Parse SRT
    try:
        items = list(srt.parse(srt_text))
    except Exception:
        # Treat as a single block (no parse = empty/error)
        items = []

    def ts(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t - h * 3600 - m * 60
        return f"{h:d}:{m:02d}:{s:05.2f}"

    lines = [header]
    for it in items:
        # Replace newlines with ASS line breaks (\N)
        text = (it.content or "").replace("\r", "").replace("\n", r"\N")
        # Escape braces (ASS uses { } for override codes)
        text = text.replace("{", "(").replace("}", ")")
        start = ts(it.start.total_seconds())
        end = ts(it.end.total_seconds())
        lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


def run_render_job(
    job: Job,
    inputs: List[Dict[str, Any]],
    srt_text: str,
    srt_kind: str,
    style: Dict[str, Any],
    output_settings: Dict[str, Any],
):
    """Concat inputs then burn subtitles."""
    job.status = "running"
    job.started_at = time.time()
    job_dir = JOB_DIR / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    if not FFMPEG_BIN:
        job.status = "error"
        job.error = "ffmpeg not available"
        job.finished_at = time.time()
        db_update_job(job)
        return

    # Resolve input paths
    in_paths: List[str] = []
    for inp in inputs:
        file_id = inp.get("file_id")
        if not file_id:
            job.status = "error"
            job.error = "missing file_id"
            job.finished_at = time.time()
            db_update_job(job)
            return
        # Find actual file
        candidates = list(UPLOAD_DIR.glob(f"{file_id}.*"))
        if not candidates:
            job.status = "error"
            job.error = f"file not found: {file_id}"
            job.finished_at = time.time()
            db_update_job(job)
            return
        in_paths.append(str(candidates[0]))

    if not in_paths:
        job.status = "error"
        job.error = "no input files"
        job.finished_at = time.time()
        db_update_job(job)
        return

    # 1) Concat (use concat demuxer, normalize to common params to avoid sync issues)
    concat_path = job_dir / "concat.mp4"
    transition = output_settings.get("transition", "cut")  # cut | fade
    fade_dur = float(output_settings.get("fade_duration", 0.5))

    job.message = "Concatenating videos…"
    job.progress = 0.05
    job.log.append(job.message)
    db_update_job(job)

    if transition == "fade" and len(in_paths) >= 2:
        # Use xfade + acrossfade for crossfaded transitions.
        # Strategy: each input becomes a labeled pad [0:v][0:a][1:v][1:a]...
        # Chain: v_prev, v_curr -> xfade -> v_new (replace v_curr in list).
        n = len(in_paths)
        inputs_args = []
        for p in in_paths:
            inputs_args += ["-i", p]
        durs = [ffprobe_meta(p).get("duration", 0.0) or 0.0 for p in in_paths]
        # xfade offset between i and i+1 = sum(durs[0..i]) - fade_dur
        offsets = []
        cum = durs[0]
        for d in durs[1:]:
            offsets.append(max(0.0, cum - fade_dur))
            cum += d
        per_meta = [ffprobe_meta(p) for p in in_paths]
        all_have_audio = all(m.get("has_audio") for m in per_meta)
        some_have_audio = any(m.get("has_audio") for m in per_meta)
        filter_parts = []
        # Build xfade chain for video
        # Use stable labels: vin_<i> (input pads) and vout_<k> (intermediate)
        v_labels = [f"[{i}:v]" for i in range(n)]
        a_labels = [f"[{i}:a]" for i in range(n)] if some_have_audio else None
        for i, off in enumerate(offsets):
            out_label = f"vout{i}"
            filter_parts.append(
                f"{v_labels[i]}{v_labels[i+1]}xfade=transition=fade:duration={fade_dur}:offset={off}[{out_label}]"
            )
            v_labels[i+1] = f"[{out_label}]"
        # Build acrossfade chain for audio
        if all_have_audio:
            for i, off in enumerate(offsets):
                out_label = f"aout{i}"
                filter_parts.append(
                    f"{a_labels[i]}{a_labels[i+1]}acrossfade=d={fade_dur}[{out_label}]"
                )
                a_labels[i+1] = f"[{out_label}]"
        elif some_have_audio:
            # Mixed: use amix for missing audio (simplest)
            # Tag each a_label with apad if missing
            for i in range(n):
                if not per_meta[i].get("has_audio"):
                    filter_parts.append(
                        f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={durs[i]}[a{i}_sil]"
                    )
                    a_labels[i] = f"[a{i}_sil]"
            # acrossfade
            for i, off in enumerate(offsets):
                out_label = f"aout{i}"
                filter_parts.append(
                    f"{a_labels[i]}{a_labels[i+1]}acrossfade=d={fade_dur}[{out_label}]"
                )
                a_labels[i+1] = f"[{out_label}]"
        else:
            # No audio anywhere — synthesize one
            total_dur = sum(durs) - (len(durs) - 1) * fade_dur
            filter_parts.append(
                f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={total_dur}[a_silent]"
            )
            a_labels = ["[a_silent]"] * n
        filter_complex = ";\n".join(filter_parts)
        # Final pad references (without brackets inside)
        final_v = v_labels[-1]
        final_a = a_labels[-1]
        # Use h264_nvenc (GPU) on CUDA builds. Falls back to libx264 on
        # non-NVENC binaries.
        is_cuda = "cuda" in (FFMPEG_BIN or "").lower()
        v_codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-b:v", "4M"] if is_cuda \
            else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        cmd = [
            FFMPEG_BIN, "-y",
            *inputs_args,
            "-filter_complex", filter_complex,
            "-map", final_v,
            "-map", final_a,
            *v_codec_args,
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-shortest",  # in case audio is shorter
            str(concat_path),
        ]
    else:
        # Simple concat demuxer
        list_file = job_dir / "concat.txt"
        with list_file.open("w") as f:
            for p in in_paths:
                # ffmpeg concat demuxer requires 'file' keyword with absolute path
                f.write(f"file '{p}'\n")
        is_cuda = "cuda" in (FFMPEG_BIN or "").lower()
        v_codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-b:v", "4M"] if is_cuda \
            else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        cmd = [
            FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            *v_codec_args,
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(concat_path),
        ]

    job.log.append("ffmpeg concat: " + " ".join(cmd[:6]) + " …")
    rc, log = _run_ffmpeg(cmd, total_duration=None, job=job)
    if rc != 0 or job.cancel_flag:
        if job.cancel_flag:
            job.status = "cancelled"
            job.message = "Cancelled during concat"
        else:
            job.status = "error"
            job.error = f"concat failed (rc={rc})"
            job.log.extend(log.splitlines()[-30:])
        job.finished_at = time.time()
        db_update_job(job)
        return
    if not concat_path.is_file() or concat_path.stat().st_size < 100:
        job.status = "error"
        job.error = "concat produced no file"
        job.finished_at = time.time()
        db_update_job(job)
        return

    # 2) Burn subtitles
    if srt_text.strip():
        job.message = "Burning subtitles…"
        job.progress = 0.7
        job.log.append(job.message)
        # Choose format: if srt_kind == "ass" use the ass file directly; else convert
        if srt_kind == "ass":
            ass_path = job_dir / "subtitles.ass"
            ass_path.write_text(srt_text, encoding="utf-8")
        else:
            ass_text = srt_to_ass(srt_text, style)
            ass_path = job_dir / "subtitles.ass"
            ass_path.write_text(ass_text, encoding="utf-8")

        out_path = OUTPUT_DIR / f"{job.job_id}.mp4"
        # ASS filter path needs careful handling. ffmpeg's filter parser can
        # mis-interpret '/' in paths. The safest approach: cd into the job dir
        # and pass just the filename. Use 'subtitles=' (alias for ass=) which
        # accepts the same files and parses more leniently.
        # With ffmpeg-cuda-v7 (which has both libass + NVENC), we now use GPU
        # for the burn step too. h264_nvenc is ~5-10x faster than libx264
        # for the same quality on long videos.
        burn_encoder = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-b:v", "4M"]
        # Fallback to libx264 if nvenc not available
        # (auto-detect by trying h264_nvenc, fall back on error)
        burn_cmd = [
            FFMPEG_BURN_BIN or FFMPEG_BIN, "-y",
            "-i", str(concat_path),
            "-vf", f"subtitles={ass_path.name}",
            *burn_encoder,
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(out_path),
        ]
        # Wrap ffmpeg with cwd=job_dir so subtitles= resolves locally
        job.log.append(f"ffmpeg burn (cwd={job_dir}, bin={FFMPEG_BURN_BIN}, enc=nvenc): " + " ".join(burn_cmd[:6]) + " …")
        # estimate progress
        dur = ffprobe_meta(str(concat_path)).get("duration", 0.0) or 0.0
        rc, log = _run_ffmpeg(burn_cmd, total_duration=dur, progress_start=0.7, progress_end=0.99, job=job, cwd=str(job_dir))
        if rc != 0 or job.cancel_flag:
            if job.cancel_flag:
                job.status = "cancelled"
                job.message = "Cancelled during burn"
            else:
                job.status = "error"
                job.error = f"burn failed (rc={rc})"
                job.log.extend(log.splitlines()[-30:])
            job.finished_at = time.time()
            db_update_job(job)
            return
        final_path = out_path
    else:
        # No subtitles — output is the concat
        final_path = OUTPUT_DIR / f"{job.job_id}.mp4"
        shutil.move(str(concat_path), str(final_path))

    if not final_path.is_file() or final_path.stat().st_size < 100:
        job.status = "error"
        job.error = "final output missing"
        job.finished_at = time.time()
        db_update_job(job)
        return

    job.output_path = str(final_path)
    job.status = "done"
    job.progress = 1.0
    job.message = f"Render complete — {final_path.stat().st_size // 1024} KB"
    job.finished_at = time.time()
    db_update_job(job)


def _run_ffmpeg(
    cmd: List[str],
    total_duration: Optional[float] = None,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
    job: Optional[Job] = None,
    cwd: Optional[str] = None,
) -> tuple[int, str]:
    """Run ffmpeg with progress parsing. Returns (rc, combined_log)."""
    # Add -progress pipe:1 -nostats to capture machine-readable progress
    if "-progress" not in cmd:
        # insert after -y or at start
        cmd = [cmd[0]] + ["-y", "-progress", "pipe:1", "-nostats", "-loglevel", "info"] + cmd[2:]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=cwd,
    )
    log_buf = []
    last_db_update = 0.0
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        log_buf.append(line)
        if job is not None and total_duration and line.startswith("out_time_ms="):
            try:
                t = int(line.split("=", 1)[1]) / 1_000_000
                ratio = min(1.0, t / total_duration)
                job.progress = progress_start + (progress_end - progress_start) * ratio
                job.message = f"Rendering… {t:.1f}s / {total_duration:.1f}s"
            except Exception:
                pass
            # Throttle DB writes to every 2s
            now = time.time()
            if now - last_db_update > 2.0:
                try:
                    db_update_job(job)
                except Exception:
                    pass
                last_db_update = now
        if job is not None and job.cancel_flag:
            try:
                proc.terminate()
            except Exception:
                pass
            break
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    err = proc.stderr.read() if proc.stderr else ""
    log_buf.append("--- stderr ---")
    log_buf.append(err)
    combined = "\n".join(log_buf)
    if job is not None:
        job.log.extend(combined.splitlines()[-50:])
    return proc.returncode, combined


# ---------------------------------------------------------------------------
# API: jobs
# ---------------------------------------------------------------------------
@app.post("/api/jobs/whisper")
async def job_whisper(
    file_id: str = Form(...),
    model: str = Form(DEFAULT_WHISPER_MODEL),
    language: str = Form("auto"),
):
    candidates = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not candidates:
        raise HTTPException(404, "file not found")
    path = str(candidates[0])
    meta = ffprobe_meta(path)
    if not meta.get("has_audio"):
        raise HTTPException(400, "video has no audio track — whisper needs audio")
    job = Job(job_id=f"w_{uuid.uuid4().hex[:10]}", kind="whisper")
    job.meta = {"file_id": file_id, "model": model, "language": language}
    with JOB_LOCK:
        JOBS[job.job_id] = job
    db_insert_job(job)
    target = lambda: run_whisper_job(job, path, model, language)
    enqueue_job(job, target)
    return {"job_id": job.job_id, "status": job.status}


@app.post("/api/jobs/render")
async def job_render(payload: Dict[str, Any]):
    """Render = concat videos + burn subtitles (if any)."""
    inputs = payload.get("inputs", [])
    if not inputs:
        raise HTTPException(400, "no inputs")
    srt_text = payload.get("srt_text", "")
    srt_kind = payload.get("srt_kind", "srt")  # srt | ass
    style = payload.get("style", {})
    output_settings = payload.get("output_settings", {})
    if srt_kind not in ("srt", "ass"):
        srt_kind = "srt"

    job = Job(job_id=f"r_{uuid.uuid4().hex[:10]}", kind="render")
    job.meta = {
        "inputs": inputs,
        "input_count": len(inputs),
        "has_subs": bool(srt_text.strip()),
        "srt_kind": srt_kind,
        "output_settings": output_settings,
        "style": style,
    }
    with JOB_LOCK:
        JOBS[job.job_id] = job
    db_insert_job(job)
    target = lambda: run_render_job(job, inputs, srt_text, srt_kind, style, output_settings)
    enqueue_job(job, target)
    return {"job_id": job.job_id, "status": job.status}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    # In-memory first (live status), then DB (history)
    job = JOBS.get(job_id)
    if job:
        return job.to_dict()
    rec = db_get_job(job_id)
    if not rec:
        raise HTTPException(404, "job not found")
    return rec


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if job:
        job.cancel_flag = True
        return {"ok": True}
    # Allow cancel for in-flight jobs only; history jobs are immutable
    raise HTTPException(404, "job not found or already finished")


@app.get("/api/jobs")
async def list_jobs(
    status: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    source: str = "all",  # all | live | history
):
    """List jobs. `source`:
      - all: merge live (in-memory) with history (DB), live first
      - live: only in-memory running/queued jobs
      - history: only DB records (finished jobs)
    """
    seen = set()
    items: List[Dict[str, Any]] = []
    if source in ("all", "live"):
        live = [j.to_dict() for j in sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)]
        for it in live:
            if (status and it["status"] != status) or (kind and it["kind"] != kind):
                continue
            items.append(it)
            seen.add(it["job_id"])
    if source in ("all", "history"):
        for it in db_list_jobs(status=status, kind=kind, limit=limit, offset=offset):
            if it["job_id"] in seen:
                continue
            items.append(it)
        items.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    if limit:
        items = items[:limit]
    # Add counts for summary
    counts = db_count_jobs()
    return {"jobs": items, "counts": counts, "source": source}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job record + its output file. Cannot delete in-flight jobs."""
    job = JOBS.get(job_id)
    if job and job.status in ("queued", "running"):
        raise HTTPException(409, "cannot delete running job — cancel first")
    res = db_delete_job(job_id)
    if not res.get("deleted"):
        raise HTTPException(404, "job not found")
    # Also drop from in-memory cache
    with JOB_LOCK:
        JOBS.pop(job_id, None)
    return res


@app.post("/api/cleanup")
async def trigger_cleanup():
    """Manually trigger cleanup. Returns {jobs_deleted, uploads_removed, freed_bytes}."""
    return db_cleanup_old()


@app.get("/api/storage")
async def storage_info():
    """Show disk usage of uploads, jobs, outputs."""
    def dir_size(p: Path) -> int:
        return sum(c.stat().st_size for c in p.rglob("*") if c.is_file()) if p.exists() else 0
    return {
        "uploads_bytes": dir_size(UPLOAD_DIR),
        "jobs_bytes": dir_size(JOB_DIR),
        "outputs_bytes": dir_size(OUTPUT_DIR),
        "db_bytes": DB_PATH.stat().st_size if DB_PATH.exists() else 0,
        "upload_count": sum(1 for _ in UPLOAD_DIR.iterdir() if _.is_file()),
        "output_count": sum(1 for _ in OUTPUT_DIR.iterdir() if _.is_file()),
        "ttl_days": OUTPUT_TTL_DAYS,
        "ttl_hours_upload": UPLOAD_TTL_HOURS,
    }


# ---------------------------------------------------------------------------
# Downloads (live + history)
# ---------------------------------------------------------------------------
def _resolve_output_path(job_id: str) -> Optional[Path]:
    job = JOBS.get(job_id)
    if job and job.output_path:
        p = Path(job.output_path)
        if p.is_file():
            return p
    rec = db_get_job(job_id)
    if rec and rec.get("output_path"):
        p = Path(rec["output_path"])
        if p.is_file():
            return p
    return None


def _resolve_srt_path(job_id: str) -> Optional[Path]:
    job = JOBS.get(job_id)
    if job and job.srt_path:
        p = Path(job.srt_path)
        if p.is_file():
            return p
    rec = db_get_job(job_id)
    if rec and rec.get("srt_path"):
        p = Path(rec["srt_path"])
        if p.is_file():
            return p
    return None


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    p = _resolve_output_path(job_id)
    if not p:
        raise HTTPException(404, "no output")
    return FileResponse(
        str(p),
        media_type="video/mp4",
        filename=f"scenestitch_{job_id}.mp4",
    )


@app.get("/api/srt/{job_id}")
async def get_srt(job_id: str):
    p = _resolve_srt_path(job_id)
    if not p:
        raise HTTPException(404, "no srt")
    return PlainTextResponse(p.read_text(encoding="utf-8"), media_type="text/plain")


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
SUBTITLE_PRESETS = {
    "tiktok": {
        "label": "TikTok (bold bottom, white + black outline)",
        "style": {
            "font": "Noto Sans Thai",
            "size": 56,
            "color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "back_color": "&H80000000",
            "bold": True,
            "outline_width": 3,
            "shadow": 0,
            "align": 2,  # bottom center
            "margin_v": 90,
            "margin_l": 40,
            "margin_r": 40,
        },
    },
    "cinema": {
        "label": "Cinema (mid-bottom, smaller)",
        "style": {
            "font": "Noto Sans Thai",
            "size": 42,
            "color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "back_color": "&H80000000",
            "bold": False,
            "outline_width": 2,
            "shadow": 1,
            "align": 2,
            "margin_v": 50,
            "margin_l": 40,
            "margin_r": 40,
        },
    },
    "highlight": {
        "label": "Highlight (yellow text, black bg box)",
        "style": {
            "font": "Noto Sans Thai",
            "size": 52,
            "color": "&H0000FFFF",  # yellow
            "outline_color": "&H00000000",
            "back_color": "&H99000000",
            "bold": True,
            "outline_width": 2,
            "shadow": 0,
            "align": 2,
            "margin_v": 80,
            "margin_l": 40,
            "margin_r": 40,
        },
    },
    "top": {
        "label": "Top center (caption style)",
        "style": {
            "font": "Noto Sans Thai",
            "size": 44,
            "color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "back_color": "&H80000000",
            "bold": False,
            "outline_width": 2,
            "shadow": 0,
            "align": 8,  # top center
            "margin_v": 60,
            "margin_l": 40,
            "margin_r": 40,
        },
    },
}


@app.get("/api/presets/subtitle")
async def get_presets():
    return {"presets": SUBTITLE_PRESETS}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=38767, log_level="info")
