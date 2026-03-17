# Presentation Guide — Medical Image Classification System
### Technical Judges · Video Demo + Live Demo · 10 Minutes
> Chest X-Ray Pneumonia Detection Platform

---

## Table of Contents

1. [The Mindset Shift for Technical Judges](#1-the-mindset-shift-for-technical-judges)
2. [Exact 10-Minute Script](#2-exact-10-minute-script)
3. [Screen-by-Screen Breakdown](#3-screen-by-screen-breakdown)
4. [Live Demo Guide](#4-live-demo-guide)
5. [Likely Q&A — Questions and Answers](#5-likely-qa--questions-and-answers)
6. [Video Recording Checklist](#6-video-recording-checklist)
7. [Live Demo Checklist](#7-live-demo-checklist)
8. [If Things Go Wrong](#8-if-things-go-wrong)

---

## 1. The Mindset Shift for Technical Judges

Technical judges are **not** impressed by what you built. They are impressed by **why you made each decision** and **what you know about the limits of your own work**. Every senior engineer in that room has built something similar. What separates you is showing you thought like an engineer, not a student who followed a tutorial.

Their internal checklist while watching you:

- Does this person understand *why* DenseNet121 and not just "I used a pretrained model"?
- Do they know what can go wrong in production?
- Did they actually solve the class imbalance or just ignore it?
- Is the Grad-CAM mathematically understood or just plugged in?
- Would this actually run in the real world?

### The One Rule

> **Lead with impact. Follow with engineering. Close with honesty.**

Every section maps to this rule. Impact first — judges decide in the first 60 seconds whether they're interested. Engineering second — now they want to know how. Honesty last — this is what they remember after you leave the room.

---

## 2. Exact 10-Minute Script

---

### Minute 0:00 – 1:00 — The Hook

Open with one sharp statement. No *"hello my name is, today I will present."*

> *"Pneumonia kills 2.5 million people annually. A radiologist under fatigue misses roughly 1 in 4 cases. We built a full-stack AI screening platform that classifies a chest X-ray in under 400 milliseconds, tells the doctor exactly which region triggered the decision, and logs every prediction to a database for medical audit. Ten minutes — let me show you how it works and why we built it the way we did."*

**Why this works:** You stated the problem, the solution, the latency number, the explainability feature, and the production concern in 30 seconds. They already know you're not a beginner.

---

### Minute 1:00 – 3:30 — The Video Demo

Play the pre-recorded video. Three parts, one continuous clip.

**Part 1 — Normal case (30 seconds)**

Narrate while it plays:

> *"Normal case — diffuse activation, no concentrated hot region, confidence 88%, 310 milliseconds on CPU."*

**Part 2 — Pneumonia case (45 seconds)**

> *"Pneumonia case. The model is attending to the left lower lobe — exactly where bacterial pneumonia consolidates. This is not a post-hoc label. This is gradient-weighted activation from features.norm5, the final dense block before global average pooling. A radiologist can look at this and verify whether the model is attending to the right anatomy."*

**Part 3 — API proof (15 seconds)**

> *"Production REST API. Every prediction also writes to a database — filename, confidence, inference time, MD5 hash of the image. Audit trail for medical compliance."*

> **The Grad-CAM heatmap on a PNEUMONIA X-ray is your strongest moment in the entire presentation. Hold it on screen for at least 3 seconds. Narrate it. Let it land.**

---

### Minute 3:30 – 5:30 — Architecture + Three Key Decisions

Show one clean architecture diagram:

```
X-ray (JPEG/PNG)
    ↓
React + Vite  (drag-and-drop, 16MB limit, MIME validation)
    ↓  POST /api/predict  multipart/form-data
FastAPI + Uvicorn  (async ASGI)
    ├── ImagePreprocessor  →  Resize 224×224, ImageNet normalize  →  [1,3,224,224] tensor
    ├── InferenceService (Singleton)  →  DenseNet121  →  softmax  →  label + confidence
    ├── GradCAMService  →  forward+backward hooks on features.norm5  →  base64 PNG
    └── SQLAlchemy ORM  →  prediction_logs  →  SQLite (dev) / PostgreSQL (prod)
    ↓
JSON response  →  ResultCard + GradCamViewer
```

Then hit exactly **three engineering decisions** — not a list of technologies, actual decisions with reasoning:

---

**Decision 1 — Why DenseNet121, not ResNet or EfficientNet (45 seconds)**

> *"DenseNet's dense connections mean every layer has direct access to gradients from all subsequent layers. On a small medical dataset — 5,216 training images — this matters because it significantly reduces the vanishing gradient problem during fine-tuning. ResNet50 has more parameters but fewer skip connections per layer. EfficientNet is faster but compound scaling doesn't benefit us at 224×224 with a 2-class problem. We kept the model at 7 million total parameters, 2.4 million trainable — intentional. It trains in 25 minutes on a T4 GPU, not hours."*

---

**Decision 2 — Class imbalance handling (30 seconds)**

> *"The dataset is 3,875 PNEUMONIA versus 1,341 NORMAL — roughly 3:1. A naive model learns to predict pneumonia for everything and gets 74% accuracy without learning anything useful. We used WeightedRandomSampler with inverse frequency weights. Every batch is statistically balanced. We did not oversample because duplicating NORMAL images adds no new information and risks overfitting on those specific scans."*

---

**Decision 3 — The Grad-CAM implementation detail (45 seconds)**

> *"Standard Grad-CAM registers a backward hook on the target module. DenseNet uses inplace ReLU operations that corrupt the gradient before the hook fires. Our fix: inside the forward hook, we register the backward hook directly on the output tensor — `output.register_hook(self._save_gradient)` — which captures the gradient before any inplace mutation happens. Our initial heatmaps were returning all-zero gradients on DenseNet. We traced the inplace ReLU through the computational graph to find this. It's not in most tutorials."*

> **This moment wins with technical judges.** You just demonstrated you debugged a non-obvious PyTorch internals problem. That is more impressive than any accuracy number.

---

### Minute 5:30 – 7:00 — Numbers + Honest Evaluation

Show confusion matrix and ROC curve side by side.

| Metric | Value |
|---|---|
| Test Accuracy | ~91–93% |
| ROC-AUC | ~97–98% |
| Pneumonia Recall (Sensitivity) | ~95–98% |
| Normal Precision | ~88–92% |
| Inference Time (CPU) | ~300–400ms |

Say this — this is the moment that separates you from every other presenter:

> *"The metric we optimized for is recall on PNEUMONIA, not accuracy. A false negative — telling a pneumonia patient they're normal — is clinically worse than a false positive that sends a healthy patient for a follow-up. Our scheduler uses ReduceLROnPlateau on validation loss, early stopping at patience 5, and we select checkpoints by ROC-AUC not accuracy. The validation set is 16 images — that's a Kaggle dataset limitation, not ours. We use the 624-image test set for all reported numbers. We never touched the test set during training or hyperparameter tuning."*

> **The last two sentences matter enormously.** Most student projects accidentally contaminate their test set or report validation metrics as test metrics. Saying this out loud tells the judges you know the rules.

---

### Minute 7:00 – 8:30 — Production Architecture Choices

> *"Three production decisions worth explaining:"*

**Docker multi-stage build:**
> *"Builder stage installs all Python deps including torch — around 250MB. Runtime stage copies only site-packages, keeping the image lean. OpenCV system dependencies — libgl1-mesa-glx — are installed separately at runtime because headless OpenCV needs them but they don't belong in the build layer."*

**Singleton InferenceService:**
> *"Loading an 85MB model from disk adds ~400ms to every request. The singleton loads once at startup via FastAPI's lifespan context manager. Every subsequent prediction is pure memory — no disk I/O in the hot path."*

**SQLite to PostgreSQL with zero code change:**
> *"Pydantic BaseSettings reads DATABASE_URL from the environment. Local dev defaults to SQLite. Docker Compose passes a PostgreSQL URL. SQLAlchemy handles both — the application code never changes. Same binary, different environment, different database."*

---

### Minute 8:30 – 9:30 — Known Limitations

Say these before the judges ask. This is the section that earns the most respect from technical audiences.

> *"Three things we know need fixing before this goes anywhere near a hospital:"*

> *"First — GradCAMService stores gradients as instance state. Under concurrent requests, request B's backward pass can overwrite request A's gradients. The fix is per-request local variables or a threading lock. We know this, we haven't fixed it yet."*

> *"Second — the Kaggle val split is 16 images. ROC-AUC of 1.0 on 16 images is meaningless. Real validation should be a 20% stratified split from the training set — around 1,000 images."*

> *"Third — this is binary classification. Bacterial and viral pneumonia have different treatment protocols. Multi-class classification distinguishing bacterial vs viral would be the logical next step clinically."*

> **Listing your own bugs wins the room.** Every judge knows software has bugs. The question is whether you know about them. Engineers who know their failure modes are engineers you can trust.

---

### Minute 9:30 – 10:00 — Close

One strong closing sentence. No summary. No "thank you for listening."

> *"We built this end-to-end — dataset pipeline, transfer learning, explainability, REST API, containerized deployment — because a model that only lives in a notebook helps nobody. The architecture is production-shaped. The only thing missing is a radiologist to validate the heatmaps clinically. Happy to go deeper on any layer."*

> **The last sentence invites Q&A on your terms.** You named the layers they can ask about.

---

## 3. Screen-by-Screen Breakdown

| Timestamp | What Is Visible on Screen |
|---|---|
| 0:00 – 1:00 | Project title card only — your voice, no visuals to distract |
| 1:00 – 3:30 | Video demo playing full screen — no slides, no overlays |
| 3:30 – 5:30 | Architecture flow diagram |
| 5:30 – 7:00 | Confusion matrix + ROC curve side by side |
| 7:00 – 8:30 | Three bullet points — one per production decision |
| 8:30 – 9:30 | "Known Limitations" slide — three points, nothing else |
| 9:30 – 10:00 | Architecture diagram again or blank screen |

---

## 4. Live Demo Guide

> Use this if you have time after the video demo or if judges ask to see the system running live.

### Setup Before You Walk In

Run through this **at least 30 minutes before** the presentation starts:

```bash
# Terminal 1 — Backend
source venv/bin/activate           # Windows: .\venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend
npm run dev
```

Confirm you see:
```
Model loaded on cpu: densenet121 v1.0
Server ready!
```

If the model is not at `ml/saved_models/best_model_densenet121.pth` the server starts but every prediction will 500. Check this before you walk in.

---

### Browser Tabs to Have Open and Ready

| Tab | URL | Purpose |
|---|---|---|
| Tab 1 | `http://localhost:5173` | Main app — this is your primary demo tab |
| Tab 2 | `http://localhost:8000/docs` | Swagger UI — shows the API is real |
| Tab 3 | `http://localhost:8000/api/health` | Quick proof model is loaded |
| Tab 4 | `http://localhost:8000/api/predictions` | Shows the DB audit trail after uploads |

Set browser zoom to **125%**. Everything must be readable on a projector without squinting.

---

### The Three X-Ray Images to Prepare

Have these saved to your Desktop or a folder you can open instantly:

| File | What It Should Show | Why You Need It |
|---|---|---|
| `normal_clear.jpeg` | Healthy lungs, clean scan | Opens green — shows NORMAL path works |
| `pneumonia_clear.jpeg` | Obvious consolidation, lower lobe | Opens red — Grad-CAM lights up the infected region |
| `pneumonia_borderline.jpeg` | Subtle infiltrate, lower confidence | Shows the model isn't just pattern-matching on obvious cases |

> Get these from the test set at `ml/data/raw/test/NORMAL/` and `ml/data/raw/test/PNEUMONIA/`. Pick images visually — open them in an image viewer first and choose ones where the pathology is clearly visible to the human eye.

---

### Live Demo Flow — Step by Step

#### Step 1 — Health Check (20 seconds)

Switch to Tab 3 (`/api/health`). Show the JSON:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "DenseNet121",
  "device": "cpu"
}
```

Say:
> *"Model is loaded, running on CPU, ready to serve. This endpoint is also what Docker uses as the HEALTHCHECK before the container is marked ready."*

---

#### Step 2 — Upload Normal X-Ray (45 seconds)

Switch to Tab 1. Drag `normal_clear.jpeg` into the upload zone.

While it processes (2–4 seconds):
> *"Multipart upload, MIME validated client-side by react-dropzone, size checked server-side before the image is decoded."*

When result appears:
> *"NORMAL, 87% confidence, 340 milliseconds. Grad-CAM is diffuse — no focal region of high activation. The model doesn't know where to look because there's nothing pathological to find."*

Point at the confidence bar. Point at the inference time. Point at the heatmap.

---

#### Step 3 — Upload Pneumonia X-Ray (60 seconds)

Drag `pneumonia_clear.jpeg` into the upload zone.

When result appears — **pause here before speaking**. Let the red heatmap sit on screen for 2 full seconds in silence. Then:

> *"PNEUMONIA, 97% confidence. Look at the Grad-CAM. The activation is concentrated in the lower lobe — that's the anatomically correct region for bacterial pneumonia consolidation. This is what separates this from a black-box classifier. The model is attending to the right thing for the right reason. A radiologist can look at this overlay and confirm or reject it."*

Then point at the confidence bar:
> *"97% confidence. The model is certain. When we see borderline cases the confidence drops to 60–70% — the system is calibrated, not overconfident."*

---

#### Step 4 — Upload Borderline Case (30 seconds)

Drag `pneumonia_borderline.jpeg` into the upload zone.

When result appears:
> *"Here's a harder case. Lower confidence — maybe 68%. The Grad-CAM shows a smaller, less defined region. This is exactly the case where you'd want a radiologist to make the final call. The system doesn't pretend to be certain when it isn't."*

---

#### Step 5 — Show the Audit Trail (20 seconds)

Switch to Tab 4 (`/api/predictions`). Show the JSON:

```json
{
  "total": 3,
  "items": [
    {
      "id": 3,
      "filename": "pneumonia_borderline.jpeg",
      "prediction": "PNEUMONIA",
      "confidence": 0.6821,
      "inference_ms": 298,
      "created_at": "2026-03-16T10:42:00"
    },
    ...
  ]
}
```

Say:
> *"Every prediction persisted. Filename, confidence, inference time, timestamp. In a hospital context this is non-negotiable — you need to know who submitted what image, what the model said, and when. The image hash using MD5 also lets you detect duplicate submissions."*

---

#### Step 6 — Show Swagger UI (20 seconds)

Switch to Tab 2 (`/docs`).

> *"Auto-generated from the FastAPI route definitions and Pydantic schemas. The API is self-documenting. Any developer can integrate this without talking to us — they read the schema and they know exactly what to POST and what they get back."*

You do not need to click anything on Swagger. Just show it exists.

---

#### Step 7 — Optional: Show the Notebook Output (if asked)

If a judge asks about training or evaluation, open `ml/reports/` and show:

- `confusion_matrix.png` — point out the low false-negative count
- `roc_curve.png` — point out the AUC
- `training_history.json` — show the val_auc climbing each epoch

> *"Training converged in 6 epochs on our local setup. Early stopping triggered before epoch 20 because the val AUC stopped improving. Best checkpoint at epoch 3 — AUC 1.0 on the 16-image val set, which is why we rely on the 624-image test set for honest evaluation."*

---

### Live Demo Timing Summary

| Step | Action | Time |
|---|---|---|
| 1 | Health check — `/api/health` | 20 sec |
| 2 | Upload NORMAL X-ray | 45 sec |
| 3 | Upload PNEUMONIA X-ray | 60 sec |
| 4 | Upload borderline case | 30 sec |
| 5 | Show audit trail — `/api/predictions` | 20 sec |
| 6 | Show Swagger UI | 20 sec |
| **Total** | | **~3.5 min** |

---

## 5. Likely Q&A — Questions and Answers

---

**"Why not use a Vision Transformer instead of DenseNet?"**

> *"ViTs need significantly more data to generalize. With 5,216 training images, the inductive bias of CNNs — specifically local feature detection — is an advantage, not a limitation. DenseNet121 with ImageNet pretraining is the standard baseline on this exact Kaggle dataset in the literature and achieves 97–98% AUC, which matches published results. ViT-Base needs roughly 14 million images to train well from scratch. Fine-tuning a pretrained ViT is possible but DenseNet121 gave us the right tradeoff between parameter count, training time, and accuracy for this dataset size."*

---

**"How do you handle input images that aren't chest X-rays?"**

> *"Currently we don't, and that's a real gap. The model will produce a confident prediction on any JPEG you give it because softmax always sums to 1. A production system needs an out-of-distribution detector — either a confidence threshold gate, a separate binary classifier that rejects non-X-ray inputs, or energy-based OOD detection. We validated MIME type and file size, but we did not validate image content."*

---

**"Why MD5 for image hashing and not SHA-256?"**

> *"Honest answer — the database column was sized at 64 characters for SHA-256 but we used MD5 which produces 32 characters. MD5 is fine for deduplication, which is the only use case here — detecting if the same image was submitted twice. It is not being used as a security hash. SHA-256 would be the correct choice for anything integrity-critical."*

---

**"What's your inference latency under concurrent load?"**

> *"We haven't load-tested it yet. Single-request CPU latency is 300–400ms. Under concurrent load there's a known thread-safety issue in GradCAMService — instance-level gradient state means request B's backward pass can overwrite request A's. In a real deployment you'd either fix the instance state with threading.Lock or run multiple Uvicorn worker processes behind the Nginx proxy, which isolates each worker's model state."*

---

**"Why FastAPI over Flask or Django?"**

> *"FastAPI gives async request handling, automatic OpenAPI documentation generation, and Pydantic validation built in — all three matter here. Flask requires separate libraries for all three and doesn't have native async support without extensions. Django is designed for full web applications — it's too heavyweight for a single-purpose inference API. FastAPI also lets us define the response schema in Python and get JSON validation, serialization, and Swagger docs from the same class definition."*

---

**"How does the Grad-CAM actually work mathematically?"**

> *"We compute the gradient of the class score with respect to the feature map activations at features.norm5 — the last dense block output after batch normalization. We global-average-pool those gradients across the spatial dimensions to get a weight per channel, then compute a weighted sum of the activation maps. Apply ReLU to keep only positive contributions — regions that pushed the score up — then normalize to 0–1 and resize to the original image dimensions. The result is a spatial importance map: high values mean that region contributed most to the predicted class."*

---

**"Why did you choose SQLite for local development?"**

> *"SQLite requires zero external dependencies — no Docker container, no connection string, no credentials. A developer clones the repo, runs uvicorn, and the database is created automatically in the project directory. Switching to PostgreSQL for production is one environment variable change — DATABASE_URL. SQLAlchemy abstracts the difference. This is a deliberate developer experience decision, not a limitation."*

---

**"What happens if the model file isn't present when the server starts?"**

> *"The server starts but load_model() raises a FileNotFoundError during the lifespan startup. Uvicorn catches it, logs the error, and the server fails to start cleanly. The health endpoint would return model_loaded: false if we got that far. In a Docker deployment the HEALTHCHECK would catch this and mark the container unhealthy before traffic is routed to it — assuming the curl bug in the Dockerfile is fixed."*

---

**"Can this be used in a real hospital?"**

> *"Not as-is. Three things need to happen first: clinical validation by radiologists on a representative patient population, regulatory approval — FDA 510k clearance in the US for a medical AI device — and the thread-safety issue needs to be fixed for concurrent production load. What we built is a technically sound proof of concept with a production-shaped architecture. The path from here to deployment is regulatory, not engineering."*

---

## 6. Video Recording Checklist

### Before You Record

- [ ] Model is loaded and **warmed up** — run one prediction manually before recording, first inference after loading is always 200–300ms slower
- [ ] Three X-ray images ready on Desktop: `normal_clear.jpeg`, `pneumonia_clear.jpeg`, `pneumonia_borderline.jpeg`
- [ ] Browser zoom set to **125%**
- [ ] Dark mode enabled on browser if available — Grad-CAM jet colormap pops more on dark backgrounds
- [ ] All other browser tabs closed — only the app tab visible
- [ ] Notifications silenced — OS, browser, Slack, everything
- [ ] Screen recording software ready at 1080p minimum, 1440p preferred
- [ ] Microphone tested — no background noise, levels set

### What to Record

| Clip | Content | Target Length |
|---|---|---|
| Clip A | Upload NORMAL X-ray, result appears, Grad-CAM loads | 30 sec |
| Clip B | Upload PNEUMONIA X-ray, result appears, **hold on Grad-CAM for 3 seconds** | 45 sec |
| Clip C | `/api/health` JSON response | 10 sec |
| Clip D | `/api/predictions` showing audit trail after uploads | 15 sec |
| Clip E | Swagger UI at `/docs` | 10 sec |

### Recording Rules

- Move the cursor **slowly and deliberately** — fast mouse movement on a projector is disorienting
- **Pause 1 full second** after each result loads before moving the mouse or speaking
- The PNEUMONIA Grad-CAM frame is the most important frame in the entire video — hold it, narrate it, do not rush past it
- Record each clip separately — edit into one continuous video afterward
- Keep the total video under **2 minutes 30 seconds**

### After Recording

- [ ] Watch the full video at normal speed — check audio sync, check nothing is cut off
- [ ] Export as MP4, H.264, at the presentation venue's expected resolution
- [ ] Have the video saved in **three places**: laptop, USB drive, cloud storage
- [ ] Have a static screenshot of the Grad-CAM result as a backup image in case video fails to play

---

## 7. Live Demo Checklist

### 30 Minutes Before

- [ ] Backend running: `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`
- [ ] Frontend running: `cd frontend && npm run dev`
- [ ] Terminal output confirms: `Model loaded on cpu: densenet121 v1.0`
- [ ] Opened `http://localhost:5173` — app loads, upload zone visible
- [ ] Opened `http://localhost:8000/api/health` — confirms `model_loaded: true`
- [ ] Test-uploaded one image — confirmed prediction and Grad-CAM render correctly
- [ ] Three X-ray images on Desktop, named clearly
- [ ] Browser zoom at 125%
- [ ] Notifications silenced
- [ ] Laptop plugged in — do not run on battery during demo
- [ ] External display connected and mirroring confirmed

### Tab Order in Browser

```
Tab 1: localhost:5173          ← Start here
Tab 2: localhost:8000/docs
Tab 3: localhost:8000/api/health
Tab 4: localhost:8000/api/predictions
```

### During the Demo

- [ ] Start on Tab 1 — always return here between steps
- [ ] Do not type during the demo — all navigation by mouse only
- [ ] If a prediction takes longer than 5 seconds — say "first inference after a cold model is slightly slower, subsequent requests are cached in memory" — it's true and it's not an apology
- [ ] If Grad-CAM loads as a blank image — say "the overlay sometimes takes a moment to decode from base64, the backend returned successfully" — then refresh
- [ ] Never say "sorry" or "that's strange" — describe what is happening technically and move on

---

## 8. If Things Go Wrong

| Problem | What to Do |
|---|---|
| Backend won't start | Check if port 8000 is already in use: `lsof -i :8000` (Mac/Linux) or `netstat -ano \| findstr :8000` (Windows). Kill the process and restart. |
| Model file not found | The `.pth` file must be at `ml/saved_models/best_model_densenet121.pth`. If it's missing, show the training history JSON and the metrics screenshots from `ml/reports/` and say "model weights are gitignored due to size, here are the evaluation outputs from training." |
| Frontend won't load | Open `http://localhost:8000/docs` instead and demo the API directly through Swagger UI. Paste a base64-encoded image if needed. |
| Grad-CAM returns blank | Show `ml/reports/gradcam_sample.png` as a pre-generated example. Say "this is the output from the notebook verification run on the test set." |
| Browser crashes | Have the video demo ready as immediate fallback. Switch to it without breaking stride. |
| Laptop display issues | Have the video and all screenshots saved on a USB drive. Use the venue's computer if needed. |
| Judge interrupts during demo | Stop immediately, answer the question fully, then offer to continue the demo or move to the next section based on their preference. Never say "I'll get to that" — answer now. |

### The Backup Stack (Always Have Ready)

```
Folder: presentation_backup/
├── demo_video.mp4                    ← Full demo video
├── gradcam_pneumonia.png             ← Best Grad-CAM screenshot
├── gradcam_normal.png                ← Normal case screenshot
├── confusion_matrix.png              ← From ml/reports/
├── roc_curve.png                     ← From ml/reports/
├── architecture_diagram.png          ← Your flow diagram
└── metrics_summary.png               ← Screenshot of key numbers
```

If everything fails, open these images one by one and narrate them. A judge who sees you handle failure calmly and continue delivering content professionally will remember that more than a perfect demo.

---

## The One Thing to Remember

> The Grad-CAM heatmap on a PNEUMONIA X-ray with the red region concentrated over the infected lobe is the single most important visual in your entire presentation. Every technical decision you explain — DenseNet121, features.norm5, the inplace ReLU hook fix — should be framed as *"this is why that heatmap is accurate."* Build toward it. Let it land. Move on.
