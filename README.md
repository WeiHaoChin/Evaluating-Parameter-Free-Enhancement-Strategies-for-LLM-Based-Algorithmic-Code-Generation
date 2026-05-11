# FYP Chat UI Demo

This is a ChatGPT-style UI demo with a JavaScript frontend and an optional Node.js backend.

## Run as a dynamic app with Python backend
1. Install Python 3.11+ and create a virtual environment.
2. Open a terminal in `d:\Github\FYP`
3. Install dependencies:
   - `python -m pip install -r requirements-python.txt`
4. Run the backend server:
   - `python backend.py`
5. Open `http://127.0.0.1:8000`

## If you only want static preview
1. Open `public/index.html` in your browser.

## Notes
- The Python backend serves the frontend from `public/` and exposes `/api/chat`.
- `public/settings.html` includes a `TextGrad loop count` setting that is passed to the backend when TextGrad is enabled.
- If you choose a model other than `mock-chat:1.0`, you must provide an API key in settings.
