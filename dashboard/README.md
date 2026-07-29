VERDICT Dashboard (MVP)

This folder contains a minimal FastAPI backend and a React frontend scaffold to interact with the VERDICT pipeline.

Backend (FastAPI)
- File: dashboard/app.py
- Run: from the repository root, after installing requirements: python dashboard\app.py
- Endpoints (MVP):
  - POST /api/triage {image_path}
  - POST /api/probe {image_path}
  - GET  /api/manifest
  - POST /api/adjudicate
  - POST /api/verify

Frontend (React)
- Folder: dashboard/frontend
- To run the frontend (locally):
  1. cd dashboard/frontend
  2. npm install
  3. npm start
- The React dev server runs on http://localhost:3000 and the backend on http://localhost:8000 (CORS allowed)

Notes
- For MVP the frontend uses server-side image paths (e.g. data/example.jpg). Upload endpoints can be added later.
- Make sure Tesseract is installed and the Python environment has required packages.
