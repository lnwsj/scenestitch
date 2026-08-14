# SceneStitch 🎬

> Video stitching + subtitle burning web app.
> FastAPI + ffmpeg + faster-whisper (GPU).

**Live**: https://stitch.cutdee.com (after DNS A record added — see below)
**Server**: `110.164.146.205` (limitrack, RTX 5060 Ti 16GB)
**Port**: `38768` (proxied via Caddy)

## Features

- **Stitch multiple videos** (mp4, mov, mkv, webm) — drag-drop, reorder, cut or fade transition
- **Auto subtitle** — `faster-whisper` STT (tiny / base / small / medium / large-v3) on any uploaded audio
- **Manual SRT** — paste / upload / edit SRT with live cue count
- **4 subtitle presets** — TikTok (bold + outline), Cinema, Highlight (yellow on black), Top center caption
- **Custom style** — font, size, color, outline width, position (top/middle/bottom + left/center/right)
- **Live progress** + downloadable MP4

## Stack

| Layer | Tech |
|------|------|
| Frontend | Vanilla HTML/JS/CSS (no build step) |
| Backend | FastAPI 0.141 + uvicorn |
| Video | ffmpeg-cuda-v6 (concat) + /usr/bin/ffmpeg (libass, burn) |
| STT | faster-whisper 1.2.1, CUDA + float16 |
| Job queue | Threading + semaphore (1 concurrent, since ffmpeg is heavy) |
| Storage | Local disk (`/opt/scenestitch/data/`) |
| Proxy | Caddy with auto-LE |

## API endpoints

```
GET  /                              # Frontend
GET  /api/health                    # ffmpeg/whisper status
GET  /api/files                     # List uploaded files
POST /api/upload                    # Upload video(s)
DELETE /api/files/{file_id}         # Delete upload
GET  /files/{name}                  # Serve uploaded file
GET  /api/presets/subtitle          # Style presets
POST /api/jobs/whisper              # Start STT job
POST /api/jobs/render               # Start render (concat + burn)
GET  /api/jobs/{id}                 # Poll job
POST /api/jobs/{id}/cancel          # Cancel
GET  /api/jobs                      # Recent jobs
GET  /api/download/{id}             # Download MP4
GET  /api/srt/{id}                  # Download SRT
```

## Deploy

### One-time setup

```bash
# Create app + venv
mkdir -p /opt/scenestitch/{data/uploads,data/jobs,data/outputs,logs,static}
python3 -m venv /opt/scenestitch/venv
source /opt/scenestitch/venv/bin/activate
pip install -r requirements.txt

# Copy app
scp app.py /opt/scenestitch/
scp static/index.html /opt/scenestitch/static/

# Install service
scp scenestitch.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now scenestitch.service

# Add Caddy site
scp stitch.cutdee.com.caddy /etc/caddy/sites-enabled/
systemctl reload caddy
```

### DNS (one-time, in Cloudflare)

Add **A record**:
- name: `stitch`
- target: `110.164.146.205`
- proxy: **DNS only** (grey cloud, NOT proxied — Caddy handles TLS)

Caddy will auto-fetch Let's Encrypt cert within ~30s.

## Configuration (env vars)

| Var | Default | Notes |
|-----|---------|-------|
| `SCENESTITCH_FFMPEG` | autodetect | For concat (prefers ffmpeg-cuda-v6) |
| `SCENESTITCH_FFMPEG_BURN` | autodetect | For burn (must have libass — uses /usr/bin/ffmpeg) |
| `SCENESTITCH_WHISPER_DEVICE` | `auto` | `auto`/`cpu`/`cuda` |
| `SCENESTITCH_WHISPER_COMPUTE` | `auto` | `auto`/`int8`/`float16`/`float32` |
| `SCENESTITCH_WHISPER_MODEL` | `small` | Default model loaded on first whisper job |
| `SCENESTITCH_UPLOAD_DIR` | `/opt/scenestitch/data/uploads` | |
| `SCENESTITCH_JOB_DIR` | `/opt/scenestitch/data/jobs` | |
| `SCENESTITCH_OUTPUT_DIR` | `/opt/scenestitch/data/outputs` | |
| `SCENESTITCH_MAX_UPLOAD_MB` | `2048` | |
| `SCENESTITCH_MAX_CONCURRENT` | `1` | ffmpeg is heavy |

## Operational notes

- **Burn step** uses `/usr/bin/ffmpeg` (which has libass). The CUDA builds in `/opt/ffmpeg-cuda-vN/` don't have libass.
- **Concat step** uses `/opt/ffmpeg-cuda-v6/bin/ffmpeg` (faster on long videos). Fallback: `/usr/bin/ffmpeg`.
- **Service** runs as `sj88backup02` user on port 38768.
- **Cleanup**: outputs in `data/outputs/` accumulate. Add a cron to delete jobs older than 7 days.
- **Thai fonts**: VPS has `/usr/share/fonts/truetype/noto/NotoSansThai-{Regular,Bold}.ttf` — set `font: "Noto Sans Thai"` in style.

## Common gotchas

1. **ffmpeg "No option name near X"** — the CUDA ffmpeg doesn't have libass. Use the burn binary override.
2. **Whisper OOM on RTX 5060 Ti 16GB** — large-v3 needs ~10GB VRAM. Use `medium` or `small` for safety.
3. **Subtitle shows boxes (□)** — missing font. Install Noto Sans Thai or specify `font: "DejaVu Sans"` (limited Thai support).
4. **Audio desync after concat** — different sample rates / channels. The code normalizes to aac 192k, but if input uses unusual codecs, run a pre-normalize pass.

## Versioning

- **v1.0.0** (2026-08-14) — initial release. Concat (cut/fade), SRT burn, Whisper STT, 4 style presets, in-memory job queue.
