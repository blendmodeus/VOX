const { app, BrowserWindow, ipcMain, screen, clipboard, globalShortcut, Tray, Menu, nativeImage } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

const isDev = !app.isPackaged;

let mainWindow;
let overlayWindow;
let tray;
let apiProcess = null;
const API_PORT = 8000;
const API_URL = `http://127.0.0.1:${API_PORT}`;

// ─── Python API Backend ───────────────────────────────────────────
const ENGINE_DIR = path.resolve(__dirname, '..', 'engine');
const API_SCRIPT = path.join(ENGINE_DIR, 'vox_api.py');
const PROJECT_ROOT = path.resolve(__dirname, '..');

// Try venv first, fall back to system python3
const fs = require('fs');
const VENV_PYTHON = path.join(ENGINE_DIR, '.venv', 'bin', 'python3');
const PYTHON_PATH = fs.existsSync(VENV_PYTHON) ? VENV_PYTHON : 'python3';

function startApiServer() {
    if (apiProcess) return;

    const { execSync } = require('child_process');

    // Kill any stale process on the API port
    try {
        execSync(`lsof -ti :${API_PORT} | xargs kill -9 2>/dev/null`, { stdio: 'ignore' });
        console.log(`[VØX] Cleared stale process on port ${API_PORT}`);
    } catch (e) {
        // No process on port — that's fine
    }

    console.log(`[VØX] Starting Python API server (${PYTHON_PATH})...`);
    console.log(`[VØX] Script: ${API_SCRIPT}`);
    apiProcess = spawn(PYTHON_PATH, [API_SCRIPT], {
        env: {
            ...process.env,
            PYTHONPATH: PROJECT_ROOT,
            NUMBA_CACHE_DIR: '/tmp/numba_cache',
        },
        stdio: ['ignore', 'pipe', 'pipe'],
    });

    apiProcess.stdout.on('data', (data) => {
        console.log(`[API] ${data.toString().trim()}`);
    });

    apiProcess.stderr.on('data', (data) => {
        console.log(`[API:err] ${data.toString().trim()}`);
    });

    apiProcess.on('close', (code) => {
        console.log(`[VØX] API server exited (code=${code})`);
        apiProcess = null;
    });

    apiProcess.on('error', (err) => {
        console.error('[VØX] Failed to start API server:', err);
        apiProcess = null;
    });
}

function stopApiServer() {
    if (apiProcess) {
        console.log('[VØX] Stopping API server...');
        apiProcess.kill('SIGTERM');
        apiProcess = null;
    }
}

// ─── Windows ──────────────────────────────────────────────────────
function createMainWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        backgroundColor: '#000000',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.cjs'),
        },
        titleBarStyle: 'hiddenInset',
        show: false,
    });

    if (isDev) {
        mainWindow.loadURL('http://localhost:5173');
    } else {
        mainWindow.loadFile(path.join(__dirname, 'dist/index.html'));
    }

    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

function createOverlayWindow() {
    const { width, height } = screen.getPrimaryDisplay().workAreaSize;

    overlayWindow = new BrowserWindow({
        width: 300,
        height: 130,
        transparent: true,
        frame: false,
        alwaysOnTop: true,
        resizable: false,
        movable: true,
        skipTaskbar: true,
        hasShadow: false,
        focusable: false,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.cjs'),
        },
    });

    // Bottom-right corner
    overlayWindow.setPosition(width - 320, height - 150);
    overlayWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

    // Overlay always loads from file — it's standalone HTML, no Vite needed
    overlayWindow.loadFile(path.join(__dirname, 'visualizer.html'));

    overlayWindow.on('closed', () => {
        overlayWindow = null;
    });
}

// ─── Tray ─────────────────────────────────────────────────────────
function createTray() {
    // Use a simple template tray icon (16x16 on Mac)
    const icon = nativeImage.createFromDataURL(
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAhGVYSWZNTQAqAAAACAAFARIAAwAAAAEAAQAAARoABQAAAAEAAABKARsABQAAAAEAAABSASgAAwAAAAEAAgAAh2kABAAAAAEAAABaAAAAAAAAAEgAAAABAAAASAAAAAEAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAEKADAAQAAAABAAAAEAAAAAAiy48HAAAACXBIWXMAAAsTAAALEwEAmpwYAAABWWlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iWE1QIENvcmUgNi4wLjAiPgogICA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPgogICAgICA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIgogICAgICAgICAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyI+CiAgICAgICAgIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6T3JpZW50YXRpb24+CiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgoZXuEHAAAAeklEQVQ4Ee2SsQ2AIBBFL6CxcQDHcAZncAgHcA2XsLJxBgsNJxD/C0dijCU0/uSRu/v/gdwB0P+QiJaqWQEn2qxeYuQFIIU18xKTKIBMR01p7BYYScg5JuAeRy8AUYDG8LSFQMA5FiFoU2ubHMBu+y/5gfz6zXyy0P0BkVMXH6qL/iEAAAAASUVORK5CYII='
    );
    icon.setTemplateImage(true);

    tray = new Tray(icon);
    tray.setToolTip('VØX — Voice Interface');

    const contextMenu = Menu.buildFromTemplate([
        { label: 'VØX', enabled: false },
        { type: 'separator' },
        {
            label: 'Toggle Recording (⌘⇧Space)',
            click: () => {
                if (overlayWindow) overlayWindow.webContents.send('hotkey-toggle');
            },
        },
        {
            label: 'Show Overlay',
            click: () => {
                if (overlayWindow) overlayWindow.show();
            },
        },
        { type: 'separator' },
        {
            label: 'Settings',
            click: () => {
                // TODO: open settings window
                if (mainWindow) mainWindow.show();
            },
        },
        { type: 'separator' },
        {
            label: 'Quit VØX',
            click: () => {
                app.quit();
            },
        },
    ]);

    tray.setContextMenu(contextMenu);
}

// ─── Global Hotkey ────────────────────────────────────────────────
function registerHotkey() {
    const hotkey = 'CommandOrControl+Shift+Space';

    const registered = globalShortcut.register(hotkey, () => {
        console.log('[VØX] Hotkey triggered: toggle recording');
        if (overlayWindow) {
            overlayWindow.webContents.send('hotkey-toggle');
        }
    });

    if (!registered) {
        console.error(`[VØX] Failed to register global shortcut: ${hotkey}`);
    } else {
        console.log(`[VØX] Global shortcut registered: ${hotkey}`);
    }
}

// ─── IPC Handlers ─────────────────────────────────────────────────

// Text injection via clipboard + osascript paste
ipcMain.on('inject-text', async (event, text) => {
    console.log('[VØX] Injecting text:', text.substring(0, 80));
    const { exec } = require('child_process');

    // Save current clipboard
    const previousContent = clipboard.readText();

    // Set transcribed text
    clipboard.writeText(text);

    if (process.platform === 'darwin') {
        setTimeout(() => {
            exec('osascript -e \'tell application "System Events" to keystroke "v" using command down\'', (err) => {
                if (err) console.error('[VØX] Paste error:', err.message);
                else console.log('[VØX] ✓ Text pasted at cursor');
            });

            // Restore clipboard
            setTimeout(() => {
                clipboard.writeText(previousContent);
            }, 800);
        }, 150);
    }
});

// Forward API URL to renderer
ipcMain.handle('get-api-url', () => API_URL);

// Window controls from traffic light buttons
ipcMain.on('overlay-resize', (event, { width, height }) => {
    if (overlayWindow) {
        overlayWindow.setSize(width, height);
        // Re-center on screen
        const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;
        overlayWindow.setPosition(sw - width - 20, sh - height - 20);
    }
});

ipcMain.on('app-quit', () => {
    app.quit();
});

ipcMain.on('overlay-minimize', () => {
    if (overlayWindow) overlayWindow.hide();
});

// ─── App Lifecycle ────────────────────────────────────────────────
app.whenReady().then(() => {
    // Check/prompt for Accessibility permission (required for ⌘V text injection)
    const { systemPreferences } = require('electron');
    const trusted = systemPreferences.isTrustedAccessibilityClient(true);
    if (!trusted) {
        console.log('[VØX] ⚠ Accessibility permission needed — macOS will prompt you.');
        console.log('[VØX] Grant access in System Settings → Privacy & Security → Accessibility');
    } else {
        console.log('[VØX] ✓ Accessibility permission granted');
    }

    // Start the Python transcription API
    startApiServer();

    // Give the API a moment to boot, then create windows
    setTimeout(() => {
        createOverlayWindow();
        createTray();
        registerHotkey();
    }, 1000);

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createMainWindow();
            createOverlayWindow();
        }
    });
});

app.on('will-quit', () => {
    globalShortcut.unregisterAll();
    stopApiServer();
});

app.on('window-all-closed', () => {
    // On macOS, keep the app running in the tray
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
