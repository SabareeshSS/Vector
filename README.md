# Vector
Plan 01

│
├── docs
│   ├── README.md
│   └── LICENSE
|
├── images
│   └── ...
|
├── guiExample.html
├── guiExample.css
├── guiExample.js
├── main.js
│
├── pythonExample.py
|
├── config.json
└── package.json

Here's the plan:

package.json: Define the Electron project, dependencies, and start script.
main.js: The main Electron process. It will create the browser window and spawn the Python Flask server (rambo_backend.py) as a child process. It will also ensure the Python process is terminated when the Electron app closes.
guiExample.html: The structure of your user interface.
guiExample.css: Basic styling for the UI.
guiExample.js: The renderer process script. It will handle user interactions (button clicks, file selection), make fetch requests to the Python Flask API endpoints (/process, /query, /clear), and update the UI accordingly.
rambo_backend.py: Your existing Python script (no changes needed based on the provided code).
Remove RAMbo2.0.py: This file contains the Gradio UI, which is being replaced by Electron.

How to Run:

Make sure you have Node.js and npm installed.
Navigate to c:\Sabareesh\Vector\deskapp\Rambo_deskapp\ in your terminal.
Run npm install (if you haven't already).
Run npm start.
This will:

Start the Electron application (main.js).
main.js will spawn the Python Flask server (rambo_backend.py). You should see output from both Electron and Python in your terminal.
The Electron window will load guiExample.html.
You can now use the UI to upload files, process them, and ask questions. The JavaScript in guiExample.js will communicate with the Python backend over HTTP.


Running the Application

Make sure Ollama is running with the required models (llama3, llava-llama3:8b-v1.1-q4_0, all-minilm:l6-v2).
Install Python dependencies: pip install -r python_backend/requirements.txt (create this file based on imports).
Install Node.js dependencies: cd rambo-electron-app and run npm install.
Start the Electron app: npm start from the rambo-electron-app directory.
This will:

Launch the Electron window.
Start the Python FastAPI backend in the background (you should see logs in the console where you ran npm start).
Allow you to use the UI to select a file, process it (calling the Python /process endpoint), and ask questions (calling the Python /query endpoint).
Important Considerations & Next Steps:

Error Handling: The provided code has basic error handling. More robust handling (e.g., specific error messages, retries, better UI feedback) is needed.
Python Environment: This setup assumes Python and necessary libraries are installed globally or in an accessible environment. For distribution, you'd need to bundle Python or use tools like PyInstaller, which adds complexity.
Path Issues: Ensure paths (especially CHROMA_PERSIST_DIR) are handled correctly, potentially using absolute paths or paths relative to the Python script's location. The current setup stores Chroma data within the python_backend folder.
Temporary Files: The handling of temporary files (uploaded file, extracted PDF images) needs careful management, especially the PDF images stored in Chroma metadata. The current implementation uses temporary paths for PDF images which will break if the backend restarts between processing and querying. A persistent storage strategy for these images is recommended for reliability.
Progress Updates: The current setup only shows "Processing..." and then the final status. Implementing real-time progress would require the backend to provide progress updates (e.g., via WebSockets or Server-Sent Events) and the frontend to listen for them.
Packaging: Use tools like electron-builder or electron-packager to create distributable installers for different operating systems.
Security: Review Electron security best practices (context isolation, disabling nodeIntegration, validating IPC messages).
UI/UX: The HTML/CSS is very basic. Enhance it for a better user experience.

How it Works Now:

When you run npm start, main.js starts the Electron app and also spawns the modified python_backend/main.py script as a child process.
The Python script waits for JSON commands on its standard input.
When you click "Select Document" in the UI (renderer.js), it uses electronAPI.openFileDialog (IPC) to ask the Main process to show the dialog.
When you click "Process File", renderer.js calls await window.electronAPI.processFile(currentFilePath).
This triggers the ipcMain.handle('process-file', ...) handler in main.js.
main.js creates a unique request ID, stores a Promise associated with it, and sends a JSON command ({"command": "process", ...}) to the Python script's stdin.
The Python script receives the command, starts processing, and sends JSON status updates ({"type": "status", ...}) and eventually a final result ({"type": "process_result", ...}) or error ({"type": "error", ...}) to its standard output, including the request ID.
main.js listens to the Python script's stdout, parses the JSON responses.
Status updates are sent to the renderer via webContents.send('python-status-update', ...) for immediate display.
When the final result/error matching the request ID arrives, main.js resolves or rejects the Promise stored in pendingRequests.
The await in renderer.js completes, receiving the final result or catching the error. The UI is updated accordingly.
A similar flow happens for submitting queries via handleQuerySubmit and ipcMain.handle('submit-query', ...).
Python logs written to stderr are captured by main.js and forwarded to the renderer's log output area via webContents.send('python-error', ...).
