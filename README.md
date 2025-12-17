# XREPORT Radiological Reports Generator

# 1. Introduction
XREPORT provides a FastAPI backend and a React (Vite) frontend.

# 2. Installation
- **Supported platforms**: Windows scripted setup; manual installation on macOS/Linux/Windows with Python 3.12+ and Node 18+.

Standard setup:
1. Clone the repository and open a terminal at the repo root.
2. Install Python dependencies: `pip install .` (or `uv pip install .`).
3. Install frontend deps: `cd XREPORT/client && npm install`.

## Windows bootstrap
- Run `XREPORT\\start_on_windows.bat` to set up local runtimes and start backend + frontend.

# 4. Usage
Backend (any OS):
1. Ensure environment variables in `XREPORT/settings/.env` reflect the desired host/port.
2. From the repo root: `uvicorn XREPORT.server.app:app --host 0.0.0.0 --port 8000`.

Frontend:
1. `cd XREPORT/client`
2. Dev server: `npm run dev -- --host --port 5173` then open the printed URL.  
   Production preview: `npm run build && npm run preview`.

### Resources
- `XREPORT/server`: FastAPI app, routers, schemas, services, and configuration loaders.
- `XREPORT/client`: React/Vite frontend.
- `XREPORT/resources/database`: runtime data; `XREPORT/resources/logs`: runtime logs; `XREPORT/resources/templates`: env templates and supporting assets.

# 5. License
This project is licensed under the MIT License. See `LICENSE` for details.

