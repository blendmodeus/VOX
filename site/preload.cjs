const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('vox', {
    // Text injection
    injectText: (text) => ipcRenderer.send('inject-text', text),

    // Recording hotkey events from main process
    onHotkeyToggle: (callback) => ipcRenderer.on('hotkey-toggle', () => callback()),

    // Transcription results
    onTranscription: (callback) => ipcRenderer.on('transcription-result', (event, data) => callback(data)),

    // Get the API URL
    getApiUrl: () => ipcRenderer.invoke('get-api-url'),

    // Window controls
    resize: (width, height) => ipcRenderer.send('overlay-resize', { width, height }),
    quit: () => ipcRenderer.send('app-quit'),
    minimize: () => ipcRenderer.send('overlay-minimize'),
});
