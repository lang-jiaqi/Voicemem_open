# openai_voice_demo desktop wrapper — floating orb

Minimal Electron shell: spawns the existing `../backend/main.py` (unchanged)
as a child process and opens a **transparent, frameless, always-on-top
floating-orb window** (`../frontend/orb.html`):

```
┌─────────────┐      ╭────╮      ┌─────────────┐
│ 左脑 · 事实  │      │ 球 │      │ 右脑 · 画像  │
│ 记忆面板     │      ╰────╯      │ 面板         │
│ (实时更新)   │     字幕/状态     │ (实时更新)   │
└─────────────┘    [开始对话]     └─────────────┘
```

- The orb animates by state: idle (dim breathing) / listening (blue pulse) /
  speaking (green) / paused-because-you're-talking (amber).
- The two panels show, for every turn, exactly what `voicemem.Search()`
  retrieved: left panel = left-brain factual hits (with score bars), right
  panel = right-brain hits tagged by source (relation / emotion_trait /
  situation_pattern / profile / ...). They update in real time from the
  same `memory_hits` message the classic page uses -- the protocol carried
  the two-hemisphere split from day one, this UI just finally displays it
  the way it was designed to be seen.
- Drag the orb to move the window; the button under it starts/stops the
  session; caption line under the orb shows live transcript / reply text.
- `ORB_URL=/ npm start` loads the classic full debug page instead.

## Quick path: local orb window + REMOTE backend (5 minutes)

If the backend is already running on a dev server (as during this
project's development), you don't need Python/models/keys locally at all --
only Node.js. The orb window runs on your machine; everything else stays
on the server, reached through an SSH tunnel:

```bash
# 1. one-time: get just this folder onto your machine
scp -r user@server:/path/to/voicemem_opensource/openai_voice_demo/desktop ./voicemem-orb
cd voicemem-orb && npm install

# 2. every time: tunnel + start
ssh -N -L 8788:localhost:8788 user@server &        # keep this running
VOICE_DEMO_PORT=8788 SKIP_BACKEND=1 npm start
```

`SKIP_BACKEND=1` skips spawning a local backend and just opens the orb
against `localhost:8788` (the tunnel). Mic audio flows from your machine
through the tunnel to the server's ASR/voicemem/OpenAI stack and the voice
comes back the same way.

## Full local setup (run this on your own machine, not the remote dev server)

```bash
# 1. voicemem_opensource + this demo's own Python deps, same as running the
#    web version (see ../README.md) -- must be importable by whatever
#    python VOICEMEM_PYTHON below resolves to. Pipeline mode also needs the
#    local ASR models: `bash scripts/download_models.sh service/models`
#    from the repo root (see ../README.md).
cd .. && pip install -e ../.. && pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY at minimum

# 2. this wrapper's own (tiny) node deps
cd desktop
npm install
```

## Run

```bash
npm start
```

If `python3` on PATH isn't the environment with voicemem_opensource
installed (e.g. you use a conda env), point at it explicitly:

```bash
VOICEMEM_PYTHON=/path/to/conda/envs/voicemem/bin/python3 npm start
```

The orb UI can also be developed/tested in a plain browser without
Electron: run the backend and open `http://localhost:<port>/orb.html`
(everything works, it just renders on a normal page background instead of
floating transparently).

## Known limitations (real, not hypothetical)

- Written and syntax-checked (`node --check`) on a headless remote server
  with no attached display -- the window itself (and especially the
  transparent/frameless/drag behavior, which is notoriously
  platform-sensitive in Electron) has never actually been seen opening.
  If it doesn't come up cleanly on your machine, that's real, unverified
  territory. The underlying WS protocol/audio logic is the same code as
  the extensively live-tested browser page, so failures are most likely
  window-chrome-level, not conversation-level.
- No packaging/installer (no `electron-builder` config) -- `npm start`
  only, dev-mode.
- No tray icon, no settings window, no mic/speaker device picker.
