/**
 * Minimal Electron wrapper around openai_voice_demo: spawns the existing
 * FastAPI backend (backend/main.py, unchanged) as a child process, opens one
 * window pointed at it, kills the backend on quit. Deliberately does not
 * reuse ../../desktop/ (the separate, more elaborate Electron app that
 * bridges to Qwen/DashScope via gateway/ + a bundled voicemem-core binary,
 * decoupled from the live voicemem_opensource/voicemem source) -- this demo
 * already has its own working, directly-imported voicemem integration and
 * frontend; wrapping it as-is preserves that instead of re-plumbing it
 * through a different stack.
 *
 * This must be run on a machine with a real display, microphone, and
 * speakers -- it was written and syntax-checked on a headless remote server
 * with no attached display (verified `electron --version` runs, that
 * webPreferences/session APIs used below exist in the installed version;
 * NOT verified by actually seeing the window open, since that's impossible
 * here). Run `npm install && npm start` from this directory on your own
 * machine to actually try it.
 */
const { app, BrowserWindow, session } = require('electron')
const { spawn } = require('child_process')
const path = require('path')

const BACKEND_DIR = path.resolve(__dirname, '..', 'backend')
// The backend needs voicemem_opensource + its deps importable -- point this
// at whatever python has that installed (e.g. a conda env's python3) if
// plain `python3` on PATH isn't it.
const PYTHON_BIN = process.env.VOICEMEM_PYTHON || 'python3'
const PORT = process.env.VOICE_DEMO_PORT || '8787'
const BACKEND_READY_TIMEOUT_MS = 20000
// SKIP_BACKEND=1: don't spawn a local backend at all -- just open the orb
// window against an ALREADY-RUNNING backend at localhost:PORT (typically the
// remote dev server reached through an SSH tunnel: `ssh -L 8788:localhost:8788
// user@server`, then `VOICE_DEMO_PORT=8788 SKIP_BACKEND=1 npm start`). This is
// the 5-minute path to a real desktop orb: only Node/Electron needed locally,
// no Python/models/keys on this machine.
const SKIP_BACKEND = process.env.SKIP_BACKEND === '1'

let backendProcess = null
let mainWindow = null

function startBackend() {
  return new Promise((resolvePromise, reject) => {
    backendProcess = spawn(PYTHON_BIN, ['main.py'], {
      cwd: BACKEND_DIR,
      env: { ...process.env, VOICE_DEMO_PORT: PORT },
    })

    let settled = false
    const finish = (fn, arg) => { if (!settled) { settled = true; fn(arg) } }

    const onOutput = (data) => {
      const text = data.toString()
      process.stdout.write(`[backend] ${text}`)
      // main.py prints this exact line (backend/main.py) once uvicorn is
      // actually listening -- the real "ready" signal, not a fixed guess.
      if (/Uvicorn running on/.test(text)) finish(resolvePromise)
    }
    backendProcess.stdout.on('data', onOutput)
    backendProcess.stderr.on('data', onOutput)
    backendProcess.on('error', (err) => finish(reject, err))
    backendProcess.on('exit', (code) => {
      console.log(`[backend] exited (code ${code})`)
      backendProcess = null
      finish(reject, new Error(`backend exited before becoming ready (code ${code})`))
    })

    // Fallback in case the ready-line match above is ever missed (buffering,
    // wording drift) -- don't hang the app forever, just try loading the URL.
    setTimeout(() => finish(resolvePromise), BACKEND_READY_TIMEOUT_MS)
  })
}

function createWindow() {
  // Floating-orb layout (frontend/orb.html): a transparent, frameless,
  // always-on-top strip with the orb in the middle and the left-brain /
  // right-brain memory panels flanking it. Dragging the orb moves the
  // window (-webkit-app-region: drag on the orb column); the toggle button
  // and the panels opt out so they stay clickable/scrollable.
  mainWindow = new BrowserWindow({
    width: 1000, height: 420,
    transparent: true, frame: false, alwaysOnTop: true,
    hasShadow: false, resizable: false, fullscreenable: false,
    title: 'voicemem orb',
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
  })
  // ORB_URL=/ to get the classic full debug page instead of the orb.
  mainWindow.loadURL(`http://localhost:${PORT}${process.env.ORB_URL || '/orb.html'}`)
  mainWindow.on('closed', () => { mainWindow = null })
}

app.whenReady().then(async () => {
  // Electron's default is to prompt (and on some platforms/configs, silently
  // deny) getUserMedia requests -- explicitly allow media so the mic capture
  // this demo depends on isn't blocked by an unanswered permission prompt.
  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    callback(permission === 'media')
  })

  if (!SKIP_BACKEND) {
    try {
      await startBackend()
    } catch (err) {
      console.error('Failed to start backend:', err)
      app.quit()
      return
    }
  }
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (backendProcess) backendProcess.kill()
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  if (backendProcess) backendProcess.kill()
})
