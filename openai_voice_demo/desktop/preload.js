// Bridge for the first-run setup window (setup.html). contextIsolation is on,
// so the page can't touch Node directly — it calls window.setup.save(key),
// which forwards to main.js over IPC.
const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('setup', {
  save: (key) => ipcRenderer.invoke('setup:save', key),
})
