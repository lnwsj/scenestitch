"""Smoke test for SceneStitch backend — verifies core flow without GPU."""
import io
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:38767")
TEST_FILES = Path(os.environ.get("TEST_FILES", "/workspace/initial_scenes"))


def wait_for_server(timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{BASE}/api/health", timeout=2)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server not up at {BASE}")


def upload_files():
    files = []
    for p in sorted(TEST_FILES.glob("*.mp4")):
        files.append(("files", (p.name, p.read_bytes(), "video/mp4")))
    r = httpx.post(f"{BASE}/api/upload", files=files, timeout=120)
    r.raise_for_status()
    return r.json()["files"]


def wait_job(job_id, timeout=300):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = httpx.get(f"{BASE}/api/jobs/{job_id}", timeout=10)
        r.raise_for_status()
        j = r.json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        last = j
        time.sleep(1)
    raise RuntimeError(f"job {job_id} timed out. last={last}")


def test_health():
    h = wait_for_server()
    assert h["ok"] is True
    assert h["ffmpeg"], "ffmpeg missing"
    print(f"  ✓ health: ffmpeg={h['ffmpeg']}")


def test_upload_and_list():
    files = upload_files()
    assert len(files) == 2, f"expected 2 files, got {len(files)}"
    print(f"  ✓ uploaded {len(files)} files:")
    for f in files:
        m = f["meta"]
        print(f"     - {f['name']}  {m.get('width')}x{m.get('height')}  {m.get('duration'):.1f}s  audio={m.get('has_audio')}")
    return files


def test_concat_only(files):
    """Render = concat without subtitles."""
    payload = {
        "inputs": [{"file_id": f["file_id"]} for f in files],
        "srt_text": "",
        "srt_kind": "srt",
        "style": {},
        "output_settings": {"transition": "cut"},
    }
    r = httpx.post(f"{BASE}/api/jobs/render", json=payload, timeout=30)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"  → render job: {job_id}")
    j = wait_job(job_id, timeout=120)
    assert j["status"] == "done", f"render failed: {j}"
    # download
    out_path = f"/tmp/smoke_concat.mp4"
    with httpx.stream("GET", f"{BASE}{j['output_url']}", timeout=60) as resp:
        with open(out_path, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    sz = os.path.getsize(out_path)
    # probe
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,codec_type", "-of", "default=nw=1", out_path],
        capture_output=True, text=True
    )
    print(f"  ✓ concat done: {sz//1024} KB\n    {out.stdout.strip().replace(chr(10),' | ')}")
    return out_path


def test_concat_with_srt(files):
    """Render = concat + burn subtitles (manual SRT)."""
    # Build SRT covering the combined duration
    srt_text = """1
00:00:00,000 --> 00:00:04,500
สวัสดีครับ SceneStitch

2
00:00:04,500 --> 00:00:09,000
ฉากแรกจาก Scene 1

3
00:00:09,000 --> 00:00:14,000
ต่อด้วย Scene 2 ที่มีเสียง

4
00:00:14,000 --> 00:00:18,000
เผาซับภาษาไทยแบบ TikTok
"""
    payload = {
        "inputs": [{"file_id": f["file_id"]} for f in files],
        "srt_text": srt_text,
        "srt_kind": "srt",
        "style": {
            "font": "Noto Sans Thai",
            "size": 48,
            "color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "outline_width": 3,
            "align": 2,
            "bold": True,
        },
        "output_settings": {"transition": "fade", "fade_duration": 0.5},
    }
    r = httpx.post(f"{BASE}/api/jobs/render", json=payload, timeout=30)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"  → render-with-srt job: {job_id}")
    j = wait_job(job_id, timeout=180)
    assert j["status"] == "done", f"render-with-srt failed: {j.get('error')}"
    out_path = f"/tmp/smoke_with_srt.mp4"
    with httpx.stream("GET", f"{BASE}{j['output_url']}", timeout=60) as resp:
        with open(out_path, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    sz = os.path.getsize(out_path)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,codec_type", "-of", "default=nw=1", out_path],
        capture_output=True, text=True
    )
    print(f"  ✓ srt-render done: {sz//1024} KB\n    {out.stdout.strip().replace(chr(10),' | ')}")
    return out_path


def main():
    print(f"== SceneStitch smoke test @ {BASE} ==")
    test_health()
    files = test_upload_and_list()
    test_concat_only(files)
    try:
        test_concat_with_srt(files)
    except Exception as e:
        print(f"  ✗ srt render failed: {e}")
        return 1
    print("\n== ALL PASS ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
