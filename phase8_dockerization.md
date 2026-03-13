# Phase 8: Docker Containerization & Cloud Deployment

## Objective
Convert the local environment configurations seamlessly into modular, cross-compatible image artifacts that can execute reliably across AWS EC2 instances, Google Cloud Compute Engines, Azure VMs, or standard Windows PC networks dynamically without reliance on manual python dependency setups (`venv`) natively. Implement reproducible CI/CD execution pipeline shells (`scripts/run_training.sh`) and configure secure Reverse Proxies (`NGINX`) encapsulating the Node React artifacts gracefully.

## Files Created
1. `docker-compose.yml` - Root orchestration definitions linking PostgreSQL (`postgres:15-alpine` natively if databases are expanded locally via `.env`), the FastAPI Python `backend`, the Node `frontend`, and an internal proxy network (`nginx`). Overrides internal networking exposing ports specifically `8000:8000` internally, `80:80` externally mappings dynamically.
2. `docker/Dockerfile.backend` - Multi-stage pipeline logic compiling heavy metadata (`OpenCV / libgl1-mesa-glx libglib2.0-0`) directly atop Python 3.10-slim. Automatically caches uncompiled models/weights via static copy instructions isolating `pip install -no-cache-dir -r requirements.txt` into independent layers accelerating subsequent builds seamlessly. Boots standard internal Healthcheck verification parameters `/api/health`.
3. `docker/Dockerfile.frontend` - Translates React artifacts natively from base Node configurations into `npm run build` productions securely transferring `/app/dist` arrays to lightweight `nginx:alpine` distributions cleanly saving hundreds of megabytes dynamically during container startup sequences.
4. `docker/nginx.conf` - Maps NGINX `server` and `listen 80` boundaries. Handles memory allocations natively (20MB overrides instead of default 1MB caps) preventing high-resolution diagnostic images from failing HTTP bounds natively. Binds `/api/` dynamic requests mapping proxy paths sequentially to the inter-container `http://backend:8000` connection stream without exposing Uvicorn public footprints to external traffic configurations natively.
5. `scripts/run_training.sh` - Python wrapper logic triggering parameter loops parsing automated DensNet validations safely mapping to shell binaries (`chmod +x run_training.sh`) for cross-compatibility deployment test validations natively.

## AWS Deployment Sequences
- EC2 generation targets `Ubuntu Server 22.04 LTS`.
- Requires open TCP ports mapping internal HTTP `80` dynamically alongside `8000` endpoints natively.
- Shell overrides execute standard `git clone` protocols and direct executions mapping sequential parameters: `docker-compose up --build -d` spawning the cluster permanently on Linux runtimes effortlessly.
