/**
 * Electron wrapper for the VoiceMem floating orb.
 *
 * Two modes, chosen automatically by app.isPackaged:
 *   · dev (npm start)         — spawns ../backend/main.py via python3, exactly
 *                               as before. Needs voicemem + deps importable.
 *   · packaged (.app / .dmg)  — spawns the self-contained backend executable
 *                               bundled at Resources/backend/voicemem-backend
 *                               (built with PyInstaller, see BUILD.md). No
 *                               system Python needed on the user's Mac.
 *
 * On first launch (packaged, no key yet) it shows a small window asking for
 * the user's OPENAI_API_KEY, saved to userData/config.json and passed to the
 * backend as an env var. The AI itself (GPT reply / TTS / memory extraction)
 * still calls OpenAI over the network — "local" here means the app runs on
 * the user's machine, not that it is offline.
 */
const { app, BrowserWindow, session, ipcMain } = require('electron')
const { spawn } = require('child_process')
const fs = require('fs')
const path = require('path')

const BACKEND_DIR = path.resolve(__dirname, '..', 'backend')
const PYTHON_BIN = process.env.VOICEMEM_PYTHON || 'python3'
const PORT = process.env.VOICE_DEMO_PORT || '8787'
const BACKEND_READY_TIMEOUT_MS = 20000
// SKIP_BACKEND=1: don't spawn any backend, just open the orb against an
// already-running one at localhost:PORT (dev / remote-tunnel path).
const SKIP_BACKEND = process.env.SKIP_BACKEND === '1'
const CONFIG_PATH = path.join(app.getPath('userData'), 'config.json')

let backendProcess = null
let mainWindow = null

// ── local config (stores the user's OpenAI key) ────────────────────────────
function loadConfig() {
  try { return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')) } catch { return {} }
}
function saveConfig(cfg) {
  try { fs.writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2)) } catch (e) {
    console.error('could not save config:', e)
  }
}

// ── first-run key prompt ───────────────────────────────────────────────────
// Resolves with the entered key, or null if the user closed the window.
function promptForApiKey() {
  return new Promise((resolve) => {
    const win = new BrowserWindow({
      width: 460, height: 300, resizable: false, title: 'VoiceMem setup',
      webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true },
    })
    win.loadFile(path.join(__dirname, 'setup.html'))
    let done = false
    ipcMain.handleOnce('setup:save', (_e, key) => {
      done = true
      win.close()
      resolve((key || '').trim() || null)
    })
    win.on('closed', () => { if (!done) resolve(null) })
  })
}

// ── backend process ────────────────────────────────────────────────────────
function spawnBackend(env) {
  if (app.isPackaged) {
    // PyInstaller onedir bundle copied in via electron-builder extraResources.
    const exe = path.join(process.resourcesPath, 'backend', 'voicemem-backend')
    return spawn(exe, [], { cwd: path.dirname(exe), env })
  }
  return spawn(PYTHON_BIN, ['main.py'], { cwd: BACKEND_DIR, env })
}

function startBackend(env) {
  return new Promise((resolvePromise, reject) => {
    backendProcess = spawnBackend({ ...env, VOICE_DEMO_PORT: PORT })

    let settled = false
    const finish = (fn, arg) => { if (!settled) { settled = true; fn(arg) } }

    const onOutput = (data) => {
      const text = data.toString()
      process.stdout.write(`[backend] ${text}`)
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
    setTimeout(() => finish(resolvePromise), BACKEND_READY_TIMEOUT_MS)
  })
}

// ── orb window ─────────────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000, height: 420,
    transparent: true, frame: false, alwaysOnTop: true,
    hasShadow: false, resizable: false, fullscreenable: false,
    title: 'voicemem orb',
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
  })
  mainWindow.loadURL(`http://localhost:${PORT}${process.env.ORB_URL || '/orb.html'}`)
  mainWindow.on('closed', () => { mainWindow = null })
}

// ── startup ────────────────────────────────────────────────────────────────
app.whenReady().then(async () => {
  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    callback(permission === 'media')   // allow mic without an extra prompt
  })

  const config = loadConfig()
  const env = { ...process.env }

  if (!SKIP_BACKEND) {
    // Need an OpenAI key. Prefer env, then saved config, else ask once.
    let key = process.env.OPENAI_API_KEY || config.OPENAI_API_KEY
    if (!key) {
      key = await promptForApiKey()
      if (!key) { app.quit(); return }       // user cancelled setup
      config.OPENAI_API_KEY = key
      saveConfig(config)
    }
    env.OPENAI_API_KEY = key

    try {
      await startBackend(env)
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
