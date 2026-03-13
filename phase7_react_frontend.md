# Phase 7: React Frontend — Upload UI, Results & Grad-CAM Viewer

## Objective
Convert raw backend API interactions into a scalable browser experience, allowing doctors/users to effortlessly drag and drop X-Rays directly into their local computer safely utilizing React interfaces handling `multipart/form-data` streams and mapping GradCAM image overlays dynamically across interactive dashboard components natively. 

## Files Created
1. `frontend/` - Root React application spawned by modern Vite modules (`npm create vite@latest frontend -- --template react`).
2. `frontend/package.json` - Integrated Node Dependencies connecting standard library React tools `react-dropzone` `axios` and `react-router-dom`, hooking dynamic dependencies natively.
3. `frontend/tailwind.config.js` & `frontend/postcss.config.js` - Integrated `@tailwindcss/postcss` rendering standard vanilla class-based styling rules into the Vite bundler without conflict.
4. `frontend/.env` - Stabs `VITE_API_URL` values back into FastAPI.
5. `frontend/src/api/axiosClient.js` - `fetchHistory` and `predictImage` HTTP async await handler components packing Javascript Form Data sequentially onto post streams to the python backend.
6. `frontend/src/components/ImageUploader.jsx` - React hooks orchestrating `<input type=file>` states rendering blue UI gradients safely on dynamic drop interactions rejecting payloads `>= 16MB` or non-standard JPEG/PNG configurations natively.
7. `frontend/src/components/ResultCard.jsx` - Conditional map checks parsing binary logic (PNEUMONIA vs NORMAL) flipping DOM elements green/red intuitively via Tailwind classes based on dynamic JSON object metadata outputs safely (`result.confidence`).
8. `frontend/src/components/GradCamViewer.jsx` - Renders diagnostic Base64 Heatmaps next to Original image structures synchronously enabling human comparative verification visually.
9. `frontend/src/pages/Home.jsx` & `App.jsx` - Root Orchestration UI loops bridging all React states (`useState`, `try/catch` UI exception logs dynamically mapping Error strings effectively overriding crashes directly).

## Deployment Build State
Executing `npm run build` maps the full uncompiled JSX node-graph directly into `frontend/dist/` optimizing JavaScript clusters into high-speed browser delivery modules perfectly for NGINX scaling logic.
