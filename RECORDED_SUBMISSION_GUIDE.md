# Recorded Project Submission Guide
### Medical Image Classification System — Chest X-Ray Pneumonia Detection
> Format: Screen Recording + Narration · Duration: 10 Minutes · Submission

---

## Table of Contents

1. [Key Differences — Recorded vs Live](#1-key-differences--recorded-vs-live)
2. [Before You Record — Full Setup Checklist](#2-before-you-record--full-setup-checklist)
3. [Recording Structure — Exact 10-Minute Breakdown](#3-recording-structure--exact-10-minute-breakdown)
4. [Word-for-Word Narration Script](#4-word-for-word-narration-script)
5. [Screen-by-Screen — What to Show When](#5-screen-by-screen--what-to-show-when)
6. [Demo Walkthrough — Step by Step](#6-demo-walkthrough--step-by-step)
7. [Recording Rules](#7-recording-rules)
8. [Editing Checklist](#8-editing-checklist)
9. [Final Submission Checklist](#9-final-submission-checklist)

---

## 1. Key Differences — Recorded vs Live

Since this is a recorded submission and not a live presentation, the
rules are completely different.

| Live Presentation | Recorded Submission |
|---|---|
| One shot, handle mistakes in real time | Unlimited retakes — record until it's clean |
| Judges see your nerves | Judges only see the final cut |
| Demo crashes = visible panic | Demo crashes = stop, fix, re-record that segment |
| Pacing controlled by room energy | Pacing entirely under your control |
| Q&A forces you to think on the spot | No Q&A — your narration IS the answer |
| Judges can interrupt | Nobody interrupts — you control the full 10 minutes |

**What this means practically:**

- There is no excuse for a shaky demo, rushed narration, or a crash on screen — you can re-record every segment as many times as needed
- Every word you say is a deliberate choice — you are writing a script, not improvising
- The editing phase is as important as the recording phase
- Sound quality matters more than in a live setting — judges are watching alone on headphones or speakers, not in a noisy room
- Your cursor movements, your pacing, the 2-second pause after a result loads — all of this is intentional and controllable

---

## 2. Before You Record — Full Setup Checklist

Complete every item on this list **before** you open your screen recorder.

### System Setup

- [ ] Laptop plugged into power — never record on battery
- [ ] All background applications closed: Slack, email, browser notifications, system alerts
- [ ] Do Not Disturb / Focus mode enabled on OS
- [ ] Phone on silent and face down
- [ ] Screen resolution set to **1920×1080** minimum
- [ ] Display scaling at **100%** — higher scaling causes blur when screen recording

### Browser Setup

- [ ] Browser zoom set to **125%** — text must be readable without squinting
- [ ] Only these 4 tabs open, in this exact order:

```
Tab 1 — http://localhost:5173            (Main app)
Tab 2 — http://localhost:8000/api/health (Health check)
Tab 3 — http://localhost:8000/docs       (Swagger UI)
Tab 4 — http://localhost:8000/api/predictions (Audit trail)
```

- [ ] Browser is in light mode — Grad-CAM jet colormap is more visible on white backgrounds
- [ ] Bookmarks bar hidden — cleaner screen
- [ ] No personal bookmarks, history, or autofill visible

### Backend and Frontend Running

```bash
# Terminal 1 — start backend
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — start frontend
cd frontend
npm run dev
```

Confirm terminal output before proceeding:

```
INFO:     Loading ML model...
Model loaded on cpu: densenet121 v1.0
INFO:     Server ready!
```

- [ ] Backend running and model loaded confirmed
- [ ] Frontend running at `localhost:5173`
- [ ] `/api/health` returns `model_loaded: true`
- [ ] **Test one full prediction manually before recording** — upload an image, confirm result + Grad-CAM render correctly
- [ ] Warm-up done — first inference after load is always 200ms slower than subsequent ones

### X-Ray Images Ready

Save these to your Desktop in a folder called `demo_images/`:

| Filename | What to Pick | Where to Find It |
|---|---|---|
| `01_normal.jpeg` | Clear healthy lungs, no obvious markings | `ml/data/raw/test/NORMAL/` — pick any |
| `02_pneumonia.jpeg` | Obvious consolidation, lower lobe | `ml/data/raw/test/PNEUMONIA/` — pick one with visible opacity |
| `03_borderline.jpeg` | Subtle infiltrate, less obvious | `ml/data/raw/test/PNEUMONIA/` — pick one that looks less severe |

Open each image in an image viewer first. You want images where the
pathology is clearly visible to the human eye — the Grad-CAM will be
more convincing when viewers can see what the model is reacting to.

### Audio Setup

- [ ] Use a headset microphone or dedicated mic — laptop built-in mic picks up keyboard, fan noise, and room echo
- [ ] Record a 10-second test clip and listen back before the full recording
- [ ] Speak at a consistent distance from the mic — do not lean in and out
- [ ] Room is quiet — close windows, turn off fans if they create noise
- [ ] No background music

### Screen Recorder Settings

- [ ] Resolution: 1920×1080
- [ ] Frame rate: 30fps minimum, 60fps preferred
- [ ] Audio: record system audio OFF, microphone ON
- [ ] Output format: MP4, H.264 codec
- [ ] Test recording: record 30 seconds, play it back, confirm video and audio are both captured and in sync

---

## 3. Recording Structure — Exact 10-Minute Breakdown

| Segment | Content | Duration | What Is Shown |
|---|---|---|---|
| **Intro** | Problem, what you built, what to expect | 0:00 – 1:00 | Title card or blank screen |
| **Demo** | Full working system walkthrough | 1:00 – 4:00 | Live application |
| **Architecture** | System design and three key decisions | 4:00 – 6:30 | Architecture diagram |
| **Model & Training** | DenseNet121, transfer learning, metrics | 6:30 – 8:00 | Confusion matrix + ROC curve |
| **Production Choices** | Docker, singleton, config switching | 8:00 – 9:00 | Three bullet points |
| **Limitations & Close** | Known issues, honest assessment, close | 9:00 – 10:00 | Limitations slide or blank |

> **Record each segment as a separate clip.** Edit them together afterward.
> This way a mistake in segment 4 does not require re-recording segments 1–3.

---

## 4. Word-for-Word Narration Script

Read this. Rehearse it. Then record it in your own natural voice —
do not read robotically from the script on screen. The script is a
guide, not a teleprompter.

---

### Segment 1 — Intro (0:00 – 1:00)

> "Pneumonia kills 2.5 million people every year. A radiologist reading X-rays under fatigue misses roughly 1 in 4 cases. This project is a full-stack AI screening platform that classifies a chest X-ray image in under 400 milliseconds, highlights exactly which region of the lung triggered the diagnosis, and logs every prediction to a database for medical audit.

> I built this as an end-to-end system — dataset pipeline, transfer learning with DenseNet121, Grad-CAM explainability, a FastAPI backend, a React frontend, and Docker containerization. Let me walk you through how it works and why each major piece was built the way it was."

**Delivery note:** Calm, measured pace. No filler words. The numbers — 2.5 million, 1 in 4, 400 milliseconds — land harder when you pause half a second after each one.

---

### Segment 2 — Demo (1:00 – 4:00)

*(Full narration is in Section 6 — Demo Walkthrough. Follow that script while recording this segment.)*

---

### Segment 3 — Architecture (4:00 – 6:30)

> "Here is the full system architecture. A user uploads a chest X-ray through the React frontend — react-dropzone handles drag and drop, validates MIME type client-side, and enforces a 16 megabyte size limit.

> The file is POSTed as multipart/form-data to the FastAPI backend running on Uvicorn. The first stop is the ImagePreprocessor, which opens the image with Pillow, converts it to RGB, resizes it to 224 by 224, converts to a tensor, and normalizes using ImageNet mean and standard deviation. This gives us a 1 by 3 by 224 by 224 tensor — a single-image batch ready for the model.

> The InferenceService is a Singleton. The model loads once at server startup via FastAPI's lifespan context manager and stays in memory. Every request after that is a pure memory operation — no disk reads in the hot path. Loading an 85 megabyte model from disk on every request would add 400 milliseconds of latency. The Singleton eliminates that.

> The model runs a forward pass, applies softmax to the logits, and returns the predicted class — NORMAL or PNEUMONIA — along with the confidence score and class index.

> The GradCAMService then runs a second forward and backward pass using the same model, this time capturing gradients at the features.norm5 layer — the final dense block output after batch normalization. I'll explain the specific implementation detail shortly.

> The overlay is base64-encoded and included directly in the JSON response, so the frontend needs no second request to display it.

> Every prediction is written to a database — SQLite locally, PostgreSQL in the Docker deployment — with the filename, confidence, inference time, and an MD5 hash of the image bytes for deduplication. This is the audit trail.

> On the frontend, ResultCard renders the prediction label and confidence bar, and GradCamViewer renders the original image side by side with the heatmap overlay."

---

### Segment 4 — Model and Training (6:30 – 8:00)

> "For the model I chose DenseNet121 with ImageNet pretrained weights. The reason DenseNet works well on small medical datasets is its dense connectivity — every layer receives feature maps from all preceding layers and passes its own to all subsequent ones. This creates shorter gradient paths during backpropagation and significantly reduces the vanishing gradient problem when fine-tuning on a dataset of only 5,216 images.

> I froze all layers except the final dense block — denseblock4 — and the classifier head, which I replaced with a 1024 to 256 linear layer, ReLU, 0.4 dropout, and a final 256 to 2 output layer. This gives us 2.4 million trainable parameters out of 7.2 million total — about 33 percent. The frozen backbone extracts ImageNet features, the unfrozen denseblock4 fine-tunes for X-ray domain features, and the new head learns the binary boundary.

> The dataset has a 3 to 1 class imbalance — 3,875 PNEUMONIA versus 1,341 NORMAL in training. A naive model learns to predict pneumonia for everything and achieves 74 percent accuracy without learning anything useful. I handled this with WeightedRandomSampler using inverse frequency weights, so every training batch is statistically balanced. I chose sampling over oversampling because duplicating NORMAL images adds no new information and increases overfitting risk on those specific scans.

> Training used Adam optimizer, CrossEntropyLoss, and ReduceLROnPlateau with patience 3 and factor 0.5. Early stopping at patience 5 prevented overfitting. Best checkpoint was selected by ROC-AUC, not accuracy — because on an imbalanced dataset accuracy is a misleading metric.

> Here are the results on the 624-image test set, which was never touched during training or hyperparameter selection."

*(Point to confusion matrix and ROC curve on screen.)*

> "Test accuracy 91 to 93 percent. ROC-AUC 97 to 98 percent. Pneumonia recall — sensitivity — 95 to 98 percent. Recall is the metric that matters clinically. A false negative — telling a pneumonia patient they are normal — is more dangerous than a false positive. The system is optimized accordingly."

---

### Segment 5 — Production Choices (8:00 – 9:00)

> "Three production decisions worth explaining.

> First — Docker multi-stage build. The builder stage installs all Python packages including PyTorch, which is around 250 megabytes. The runtime stage starts fresh from python:3.10-slim, copies only the installed site-packages, and installs just the OpenCV system dependencies needed at runtime. This keeps the production image lean without sacrificing the full dependency set during installation.

> Second — Singleton InferenceService. As I mentioned, the model loads once at startup and stays in memory. This is implemented using Python's dunder new to ensure only one instance of InferenceService is ever created, and a loaded flag that short-circuits any subsequent load attempts. Every request after the first benefits from a warm model with no I/O overhead.

> Third — environment-switchable database. Pydantic BaseSettings reads DATABASE_URL from the environment file. Locally it defaults to SQLite with zero configuration required — the database file is created automatically. In Docker Compose, the environment variable points to PostgreSQL. SQLAlchemy handles both dialects. The application code never changes between environments — only the environment variable."

---

### Segment 6 — Limitations and Close (9:00 – 10:00)

> "Three things I know need improvement before this system could be used in production.

> First — GradCAMService is not thread-safe. Gradients and activations are stored as instance state. Under concurrent requests, one request's backward pass can overwrite another's gradient values. The fix is to pass gradients through the call chain as local variables, or to use a threading lock. This is a known issue.

> Second — the Kaggle validation set is 16 images. A ROC-AUC of 1.0 on 16 images is statistically meaningless. A proper validation split should be around 1,000 images drawn stratified from the training set. All reported metrics use the 624-image test set, which is held-out and untouched, but the validation signal during training was weak.

> Third — this is binary classification. Bacterial and viral pneumonia have different treatment protocols. Distinguishing between them would require a multi-class model and a differently labeled dataset. That is the natural next step clinically.

> To close — the goal of this project was to build something production-shaped, not just a notebook with an accuracy number. The pipeline goes from raw JPEG to a classified result with a heatmap overlay and a database audit entry, containerized and deployable on any machine with Docker. The engineering decisions — DenseNet121, WeightedRandomSampler, the Grad-CAM hook fix, the Singleton pattern, the environment-switchable config — each one was made for a specific reason, and I have explained each of those reasons. Thank you."

**Delivery note:** The close is the one moment you can slow down completely. Let the last sentence land. Do not rush to stop the recording — pause for 2 full seconds after "thank you" before stopping.

---

## 5. Screen-by-Screen — What to Show When

| Timestamp | What is on Screen | Notes |
|---|---|---|
| 0:00 – 1:00 | **Title card** — project name, your name, date | Plain slide or text on blank screen — no distractions while you set the context |
| 1:00 – 1:20 | **`/api/health` JSON** in browser | Quick proof the system is live before the demo starts |
| 1:20 – 2:10 | **Main app** — upload NORMAL X-ray, result appears | Stay on this screen until Grad-CAM fully loads |
| 2:10 – 3:10 | **Main app** — upload PNEUMONIA X-ray, result appears | Hold on the Grad-CAM for 3 full seconds minimum |
| 3:10 – 3:30 | **Main app** — upload borderline case | Show lower confidence score |
| 3:30 – 3:50 | **`/api/predictions`** — audit trail JSON | Scroll slowly so all three entries are visible |
| 3:50 – 4:00 | **Swagger UI `/docs`** | 10 second glance — proof of self-documenting API |
| 4:00 – 6:30 | **Architecture diagram** | Keep this on screen the entire segment — no switching |
| 6:30 – 8:00 | **Confusion matrix + ROC curve** side by side | Point cursor at relevant numbers as you name them |
| 8:00 – 9:00 | **Three production decisions** — text slide | One bullet per decision, appear as you speak each one |
| 9:00 – 10:00 | **Known Limitations** — text slide | Three points, clean, no extra content |

---

## 6. Demo Walkthrough — Step by Step

This is the most important segment of the entire recording. Follow
this exactly.

---

### Step 1 — Prove the System Is Live (20 seconds)

Switch to Tab 2 — `localhost:8000/api/health`.

Narrate:

> "Before uploading anything — the system status. Model is loaded, running on CPU, DenseNet121 version 1.0. This endpoint is also what the Docker HEALTHCHECK polls before routing traffic to the container."

Show the JSON clearly:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "DenseNet121",
  "device": "cpu"
}
```

Move cursor slowly over the response. Do not click anything. Switch back to Tab 1.

---

### Step 2 — Upload the NORMAL X-Ray (50 seconds)

Switch to Tab 1. Drag `01_normal.jpeg` slowly into the upload zone.

While it processes (2–3 seconds of silence is fine — do not fill it with nervous talking):

> "File is uploaded as multipart/form-data. MIME type validated. Size checked against the 16 megabyte server-side limit. Pillow decodes and converts to RGB. ImagePreprocessor resizes to 224 by 224 and normalizes."

When the result card appears:

> "NORMAL. Confidence — [read the number on screen]%. Inference time — [read the number] milliseconds on CPU."

When the Grad-CAM loads — **pause for 1 full second in silence before speaking:**

> "The Grad-CAM. Diffuse activation across the lung fields — no focal region of high intensity. The model has no specific anatomical area to point to because there is nothing pathological present. This is what a correct NORMAL prediction looks like from an explainability perspective."

Move cursor slowly across the heatmap as you speak.

---

### Step 3 — Upload the PNEUMONIA X-Ray (70 seconds)

This is the most important step. Take your time.

Drag `02_pneumonia.jpeg` into the upload zone.

While it processes — silence. Do not speak. Let the anticipation build.

When the result card appears:

> "PNEUMONIA. Confidence — [read the number]%. [Read inference time] milliseconds."

When the Grad-CAM loads — **stop speaking entirely for 2 full seconds.** Let the viewer look at the heatmap. Then:

> "Look at the activation. Concentrated in the lower lobe. That is anatomically correct — bacterial pneumonia consolidates predominantly in the lower lobes. This is the most important output of the entire system. It is not enough to say PNEUMONIA — a clinician needs to know where and why. The Grad-CAM answers both."

Pause again.

> "This heatmap is generated by gradient-weighted class activation mapping on the features.norm5 layer — the final dense block output. We compute the gradient of the PNEUMONIA class score with respect to the feature activations, pool those gradients spatially, weight the activation channels, apply ReLU to keep only positive contributions, and normalize to 0 to 1. The result is a spatial importance map — bright means this region contributed most to the predicted class."

> "One implementation detail worth noting: DenseNet uses inplace ReLU operations that corrupt gradients before a standard module-level backward hook fires. The fix is to register the backward hook directly on the output tensor inside the forward hook — output dot register underscore hook — which captures the gradient before any inplace mutation. Our initial heatmaps were returning all-zero gradients. We traced this through the DenseNet computational graph to find the cause."

---

### Step 4 — Upload the Borderline Case (30 seconds)

Drag `03_borderline.jpeg` into the upload zone.

When result appears:

> "A harder case. Confidence drops to [read the number]% — the model is less certain. The Grad-CAM activation is smaller and less defined. This is a case where the system appropriately communicates uncertainty rather than forcing a high-confidence prediction. The output is calibrated, not overconfident."

---

### Step 5 — Show the Audit Trail (25 seconds)

Switch to Tab 4 — `localhost:8000/api/predictions`.

Scroll slowly so all three predictions are visible. Narrate:

> "Every prediction persisted to the database. Filename, prediction label, confidence score, inference time in milliseconds, and timestamp. The image is also stored as an MD5 hash for deduplication — if the same image is submitted twice, the duplicate is identifiable without storing the image itself. In a medical context this audit trail is non-negotiable."

---

### Step 6 — Show Swagger UI (15 seconds)

Switch to Tab 3 — `localhost:8000/docs`.

> "The API is self-documenting via FastAPI's automatic OpenAPI generation. Schemas, request formats, response models — all derived from the Pydantic type definitions in the codebase. Any developer can integrate this endpoint without additional documentation."

Do not click anything. This is a 15-second visual only.

---

### Demo Timing Summary

| Step | Action | Time |
|---|---|---|
| 1 | Health check | 20 sec |
| 2 | NORMAL X-ray upload | 50 sec |
| 3 | PNEUMONIA X-ray upload | 70 sec |
| 4 | Borderline case | 30 sec |
| 5 | Audit trail | 25 sec |
| 6 | Swagger UI | 15 sec |
| **Total** | | **~3 min 30 sec** |

---

## 7. Recording Rules

### Cursor Rules

- Move the cursor **slowly at all times** — fast movement on a recording is disorienting and looks nervous
- When pointing at something, hold the cursor still over it for at least 2 seconds while you speak about it
- Never move the cursor while you are not actively pointing at something specific — park it at the edge of the screen
- Do not use cursor highlighting tools — they draw attention away from the content

### Pacing Rules

- Speak **20% slower than feels natural** — recordings always sound faster on playback than they feel while recording
- Pause **1 full second** after every result loads before speaking
- Pause **2 full seconds** on the PNEUMONIA Grad-CAM before narrating it — this is your most important visual
- Never fill silence with filler words — "um", "so", "basically", "you know" — silence is fine
- If you make a mistake mid-sentence, stop completely, pause 2 seconds, and restart the sentence from the beginning — the edit point will be clean

### Segment Recording Rules

- Record each of the 6 segments as a **separate video file**
- Label them clearly: `01_intro.mp4`, `02_demo.mp4`, `03_architecture.mp4` etc.
- If you make a mistake in the middle of a segment, stop recording, take a breath, and re-record the entire segment from the beginning — do not try to patch mid-clip
- Do at least **2 takes of every segment** — always have a backup
- The demo segment specifically should have **3 takes minimum** — one mistake in the demo is easy to make and painful to edit around

### Audio Rules

- Drink water before recording — dry mouth causes clicking sounds
- Do not record immediately after eating
- Speak at a **consistent volume** throughout — do not get quieter at the end of sentences
- If a car passes, a door slams, or any background noise occurs — stop, wait for silence, re-record from the last clean sentence
- Listen back to every segment immediately after recording before moving to the next one

---

## 8. Editing Checklist

### Tools

Use any screen recording software that supports MP4 export. For editing:
- **Windows:** DaVinci Resolve (free), CapCut, or Clipchamp
- **Mac:** iMovie, DaVinci Resolve, or QuickTime basic trim
- **Any OS:** DaVinci Resolve free version handles everything needed here

### Edit Order

- [ ] Import all segment clips
- [ ] Select the best take of each segment
- [ ] Trim the start of each clip — remove the first second before you start speaking
- [ ] Trim the end of each clip — remove everything after your last word plus 1 second of silence
- [ ] Join segments in order: Intro → Demo → Architecture → Model → Production → Limitations
- [ ] Add a **0.5 second black fade** between each segment join — not a hard cut, not a long fade
- [ ] Check total duration is **between 9:30 and 10:00** — aim for exactly this range
- [ ] Watch the full joined video once at normal speed from start to finish
- [ ] Watch it again with eyes closed — audio only — check for volume inconsistencies, background noise, filler words
- [ ] If any segment has distracting audio issues, re-record that segment

### What Not to Add

- No background music — it is distracting and unprofessional in a technical submission
- No animated transitions other than the simple black fade
- No zoom effects or screen highlights added in post — if something needs to be highlighted, point at it with the cursor during recording
- No text overlays or captions — your narration covers everything
- No intro animation or logo

### Export Settings

```
Format:     MP4
Codec:      H.264
Resolution: 1920×1080
Frame rate: match your recording (30fps or 60fps)
Audio:      AAC, 44.1kHz, stereo
Bitrate:    8–12 Mbps for video
File size:  aim for under 500MB
```

---

## 9. Final Submission Checklist

### Content Check

- [ ] Total duration is between 9 minutes 30 seconds and 10 minutes exactly
- [ ] Intro states the problem clearly in the first 30 seconds
- [ ] Demo shows: NORMAL result, PNEUMONIA result with Grad-CAM held for 3 seconds, borderline case, audit trail, Swagger UI
- [ ] Architecture diagram is on screen for the full architecture segment — not switched away from mid-explanation
- [ ] Three engineering decisions are explained with reasoning, not just named: DenseNet121 choice, WeightedRandomSampler, Grad-CAM hook fix
- [ ] Metrics are quoted from the test set specifically — not validation set
- [ ] Three known limitations are stated clearly
- [ ] Close does not end with "um" or trail off — clean final sentence

### Technical Check

- [ ] Video resolution is 1920×1080
- [ ] No black bars, letterboxing, or pillarboxing
- [ ] Audio is clear throughout — no background noise, no clipping, no sections where voice drops inaudible
- [ ] Cursor is always visible when pointing at something
- [ ] All text on screen is readable — nothing is cut off at the edges
- [ ] Browser tabs are not showing personal information, other projects, or irrelevant history
- [ ] Terminal windows are not visible during the demo segments
- [ ] No notifications appear on screen at any point

### File Submission

- [ ] File is named according to submission requirements
- [ ] File format matches what was requested — typically MP4
- [ ] File size is within any stated limit
- [ ] Video plays correctly end to end on a different device before submitting
- [ ] Submitted before the deadline with time to spare — do not submit in the final hour

### Backup

Keep these files saved separately even after submission:

```
recording_backup/
├── 01_intro_take2.mp4
├── 02_demo_take3.mp4
├── 03_architecture_take1.mp4
├── 04_model_take2.mp4
├── 05_production_take1.mp4
├── 06_limitations_take2.mp4
├── final_edit_v1.mp4
└── final_edit_v2.mp4          ← submitted version
```

If the submission portal has an issue or you are asked to resubmit,
you need these files immediately. Do not delete them until after the
project is fully graded and closed.

---

## The Single Frame That Matters Most

In the entire 10-minute video, one frame carries more weight than any other.

It is the moment when the PNEUMONIA Grad-CAM heatmap appears on screen with the red activation concentrated over the lower lobe — and you hold silence for 2 seconds before explaining what you are looking at.

That frame demonstrates:

- The model is not a black box
- The prediction is anatomically grounded
- You understood the clinical context, not just the technical one
- The system is built for a human to use alongside, not to replace the radiologist

Every technical decision in the project — DenseNet121's dense connections, the inplace ReLU hook fix, the features.norm5 target layer, the ImageNet normalization, the 224×224 resize — all of it exists to make that one frame accurate and meaningful.

Build your recording around that moment. Everything before it is setup. Everything after it is explanation.
