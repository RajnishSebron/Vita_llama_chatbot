# 🌱 VitaLlama 3.1: Clinical Nutritionist LLM & Full-Stack System

[![Model](https://img.shields.io/badge/Model-Meta--Llama--3.1--8B-blue?logo=meta)](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
[![Fine-Tuning](https://img.shields.io/badge/Fine--Tuning-QLoRA%20%7C%20Unsloth-orange)](https://github.com/unslothai/unsloth)
[![Backend](https://img.shields.io/badge/Backend-FastAPI%20%7C%20Uvicorn-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Inference Accelerator](https://img.shields.io/badge/Accelerator-Tesla%20T4%20(16GB%20VRAM)-purple?logo=nvidia)](https://www.nvidia.com/)
[![Frontend](https://img.shields.io/badge/Frontend-HTML5%20%7C%20CSS3%20%7C%20Vanilla%20JS-yellow?logo=javascript)](https://developer.mozilla.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

An end-to-end, full-stack clinical AI platform featuring a fine-tuned **Llama 3.1 (8B)** model specialized in evidence-based clinical nutrition, dietetics, and metabolic health. Hosted on **Google Colab (Tesla T4 GPU)** with persistent model weights in **Google Drive**, served via **FastAPI** with non-blocking Server-Sent Events (SSE), and backed by a **Three-Tier Hybrid Memory Engine**.

---

## 📑 Table of Contents
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Three-Tier Memory Engine](#-three-tier-memory-engine)
- [Model Fine-Tuning Pipeline (Phase 1)](#-model-fine-tuning-pipeline-phase-1)
- [Backend Inference Engine (Phase 2)](#-backend-inference-engine-phase-2)
- [Frontend User Interface](#-frontend-user-interface)
- [API Reference](#-api-reference)
- [Deployment & Setup Guide](#-deployment--setup-guide)
- [Repository Structure](#-repository-structure)
- [Clinical Safety & Disclaimer](#-clinical-safety--disclaimer)

---

## 🏛 System Architecture

```mermaid
flowchart TD
    subgraph Client["Frontend Client (Browser)"]
        UI["Chat Interface (index.html)"]
        Inspector["Live Memory Inspector (app.js)"]
        ConfigBar["Tunnel Connector (localStorage)"]
    end

    subgraph Tunnel["Public Network Gateway"]
        Gateway["Ngrok Tunnel / Cloudflare Tunnel"]
    end

    subgraph ColabVM["Google Colab VM (Tesla T4 GPU - 16 GB VRAM)"]
        FastAPIApp["FastAPI REST & Streaming Server"]
        
        subgraph MemoryEngine["Three-Tier Hybrid Memory Engine"]
            Tier1["Tier 1: Normalized Query Cache (TTLCache / SHA-256)"]
            Tier2["Tier 2: Conversation Sliding Window (K = 6 turns)"]
            Tier3["Tier 3: Rolling Clinical Summary (Background Async LLM)"]
        end

        subgraph ModelCore["Inference Core"]
            FastModel["Unsloth FastLanguageModel (Llama 3.1 8B 4-bit NF4)"]
            Streamer["TextIteratorStreamer (Thread-isolated)"]
            GPULock["GPU Mutex Lock (Thread Safety)"]
        end
    end

    subgraph Storage["Persistent Cloud Storage"]
        GDrive["Google Drive (/content/drive/MyDrive/.../merged_model_drive)"]
    end

    UI -->|POST /chat (JSON)| Gateway
    Gateway -->|Forward traffic| FastAPIApp
    
    FastAPIApp --> Tier1
    Tier1 -->|Cache Hit < 10ms| FastAPIApp
    Tier1 -->|Cache Miss| Tier2
    
    Tier2 -->|Format ChatML Prompt| ModelCore
    Tier2 -->|Overflow > K messages| Tier3
    Tier3 -->|Async LLM extraction| ModelCore
    Tier3 -.->|Inject into System Prompt| Tier2

    GDrive -->|Load 16-bit / 4-bit merged weights| FastModel
    FastModel --> Streamer
    Streamer -->|SSE stream text/event-stream| FastAPIApp
    FastAPIApp -->|Chunked token events| UI
    FastAPIApp -->|Metadata & summary updates| Inspector
```

---

## 🌟 Key Features

- **Domain-Specific Fine-Tuned LLM:** Custom-trained on 4,800+ curated medical consultations filtered for clinical nutrition, macronutrient partitioning, metabolic conditions (diabetes, PCOS, IBS, hypertension), and micronutrient deficiencies.
- **Three-Tier Memory Architecture:**
  - **Tier 1:** Instant answer cache (<10ms latency) using SHA-256 normalized queries and `cachetools.TTLCache`.
  - **Tier 2:** Strict $K=6$ message sliding window keeping recent dialogue history verbatim.
  - **Tier 3:** Rolling background clinical summarization that extracts and tracks patient demographics, allergies, targets, and conditions directly inside the System Prompt.
- **Strict ChatML Prompt Formatting:** Enforces standard `<|im_start|>` and `<|im_end|>` delimiters across training and inference.
- **Non-Blocking Streaming:** Uses Hugging Face's `TextIteratorStreamer` inside Python daemon threads to stream tokens via Server-Sent Events (SSE) without blocking the FastAPI event loop.
- **Tesla T4 Optimization:** Loaded in 4-bit NormalFloat (NF4) via Unsloth, keeping active VRAM consumption under **6.0 GB / 15.0 GB**, allowing background summarization passes without CUDA OOM errors.
- **Decoupled Frontend:** Clean HTML5, CSS3, and Vanilla JavaScript architecture with no Node.js/npm dependencies, zero-config Colab tunnel connection bar, and a live Memory Inspector sidebar.

---

## 🛠 Technology Stack

### Machine Learning & LLM Core
- **Base Model:** Meta Llama 3.1 8B Instruct
- **Quantization:** 4-bit NormalFloat (NF4) via `bitsandbytes`
- **Fine-Tuning Framework:** [Unsloth AI](https://github.com/unslothai/unsloth) + Hugging Face `TRL` (`SFTTrainer`) + `PEFT` (QLoRA)
- **Tokenization:** Hugging Face `transformers` mapped to ChatML format
- **Inference Acceleration:** Native PyTorch Scaled Dot-Product Attention (SDPA) + Unsloth Triton kernels

### Backend & Serving
- **Framework:** FastAPI
- **ASGI Server:** Uvicorn (threaded with custom `asyncio` event loop)
- **Memory & Cache:** `cachetools.TTLCache`
- **Streaming:** Server-Sent Events (`StreamingResponse(media_type="text/event-stream")`)
- **Tunnels:** `pyngrok` and Cloudflare Tunnel (`cloudflared`)

### Frontend
- **Markup & Layout:** HTML5 Semantic elements
- **Styling:** CSS3 (CSS Variables, Flexbox, Grid, Glassmorphism, Responsive Media Queries)
- **Logic:** Vanilla JavaScript (ES6+, `ReadableStream`, `TextDecoder`, `AbortController`)

---

## 🧠 Three-Tier Memory Engine

Traditional conversational AI architectures either truncate chat history abruptly or run expensive context windows that quickly exceed GPU VRAM limits. VitaLlama implements a **Three-Tier Hybrid Memory Hierarchy**:

| Tier | Component | Mechanism | Target Latency | Storage / Lifecycle |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Query Cache** | SHA-256 hash of normalized text (`lower()`, stripped punctuation, whitespace collapse). Hits return pre-cached answers immediately. | `< 10 ms` | `TTLCache` (maxsize=1024, TTL=4 hours) |
| **Tier 2** | **Sliding Window** | Retains the last $K=6$ turns (3 user turns + 3 assistant turns) in raw text for immediate dialogue flow. | Real-time | In-memory session store |
| **Tier 3** | **Clinical Summary** | When turns exceed $K$, overflowing turns pass through an async background LLM extraction task. Compact bullet points update patient profile. | Background async | Injected into System Prompt |

### ChatML Injected Prompt Structure
```
<|im_start|>system
You are an empathetic, evidence-based clinical nutritionist. Provide scientifically accurate dietary, macronutrient, and micronutrient guidance. Never diagnose severe medical conditions; always refer such patients to a physician.

[ACTIVE PATIENT CLINICAL PROFILE & SUMMARY]
• Demographics: 32-year-old male, marathon runner.
• Weight/BMI: 72 kg, normoweight.
• Intolerances: Lactose intolerant.
• Targets: High carbohydrate, 130g daily protein.
Instructions: Tailor dietary & micronutrient guidance to the patient profile above.<|im_end|>
<|im_start|>user
[Prior turn from Tier 2]<|im_end|>
<|im_start|>assistant
[Prior turn from Tier 2]<|im_end|>
<|im_start|>user
What post-run recovery snack would you recommend considering my restrictions?<|im_end|>
<|im_start|>assistant
```

---

## 🧪 Model Fine-Tuning Pipeline (Phase 1)

Documented inside [`Chatbot_fine_tuning.ipynb`](Chatbot_fine_tuning.ipynb):

1. **Dataset Selection & Vectorized Filtering:**
   - Raw dataset: `lavita/ChatDoctor-HealthCareMagic-100k` (112,000+ medical dialogue pairs).
   - Regex filtering across queries and responses targeting clinical nutrition, macronutrients, vitamins, electrolytes, diabetes, celiac, IBS, and renal dietetics.
   - Cleaned forum artifacts, greetings, and disclaimers to ensure direct clinical answers.
   - Quality-filtered sample: 1,500 consultations split 90% training (1,350) and 10% validation (150).
2. **QLoRA Hyperparameters:**
   - LoRA Rank ($r$): 16
   - LoRA Alpha ($\alpha$): 16
   - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
   - Learning Rate: `2e-4` (Linear schedule with 10 warmup steps)
   - Optimizer: `adamw_8bit`
   - Gradient Accumulation Steps: 4 (Effective batch size = 8)
   - Mixed Precision: `fp16 = True` (Optimized for Tesla T4 Compute Capability 7.5)
3. **Model Export & Storage:**
   - Adapters merged directly into base weights via `model.save_pretrained_merged(save_method="merged_16bit")`.
   - Stored in Google Drive at `/content/drive/MyDrive/Nutritionist_LLM_Phase1/merged_model_drive`.

---

## ⚡ Backend Inference Engine (Phase 2)

Documented inside [`VitaLlama_Colab_Server.ipynb`](VitaLlama_Colab_Server.ipynb):

- **FastLanguageModel Integration:** Loaded via Unsloth with native 4-bit precision and `FastLanguageModel.for_inference(model)` for 2x faster token generation.
- **Concurrency & GPU Safety:** Incorporates `threading.Lock()` (`gpu_lock`) around `model.generate()`, ensuring live user chat streaming and background Tier 3 summarization tasks never collide on the single GPU.
- **Python 3.13 / Colab Compatibility:** Spawns a dedicated OS thread with its own isolated `asyncio` event loop running `loop.run_until_complete(server.serve())`, eliminating `nest_asyncio` coroutine cancellation bugs.

---

## 🖥 Frontend User Interface

Split into clean, modular files:
- [`index.html`](index.html): HTML5 semantic structure.
- [`style.css`](style.css): Dark clinical theme, responsive design, glowing status pills, and slide-out drawer for mobile devices.
- [`app.js`](app.js): Vanilla JavaScript client managing connection lifecycle, localStorage persistence, and SSE chunk parsing.

### UI Capabilities
- **Colab API Connector Bar:** Paste your Ngrok (`https://xxxx.ngrok-free.app`) or Cloudflare (`https://xxxx.trycloudflare.com`) URL directly into the UI with one-click connection verification.
- **Bypass Headers:** Automatically injects `'ngrok-skip-browser-warning': 'true'` to bypass free-tier interstitial pages.
- **Live Memory Inspector Sidebar:**
  - Real-time indicator: `⚡ Instant Cache Hit` vs `🧠 GPU Inference`
  - Active Session UUID and execution latency timer
  - Sliding window gauge ($X / 6$ active turns)
  - Live rolling clinical summary viewer
  - "Reset Session Memory" button that wipes server-side context and resets local UI state.

---

## 📡 API Reference

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "sessions": 1,
  "cache_size": 12,
  "cache_stats": { "total": 15, "hits": 3, "misses": 12 },
  "vram_alloc_gb": 5.48
}
```

### 2. Stream Chat Completion (SSE)
```http
POST /chat
Content-Type: application/json
ngrok-skip-browser-warning: true
```
**Payload:**
```json
{
  "message": "I am a 30yo runner with lactose intolerance. What should I eat for recovery?",
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "temperature": 0.3,
  "max_tokens": 512
}
```
**SSE Event Stream:**
```text
event: metadata
data: {"source": "gpu", "cache_hit": false, "session_id": "...", "clinical_summary": "...", "window_turns": 1}

event: ttft
data: {"ttft_ms": 142.5}

event: token
data: {"token": "For "}

event: token
data: {"token": "post-run recovery..."}

event: done
data: {"done": true, "source": "gpu", "duration_ms": 1240.2, "clinical_summary": "..."}
```

### 3. Inspect Session Memory
```http
GET /session/{session_id}
```

### 4. Reset Session Memory
```http
POST /session/{session_id}/reset
```

### 5. Cache Metrics
```http
GET /cache/stats
```

---

## 🚀 Deployment & Setup Guide

### Step 1: Open Server in Google Colab
1. Upload [`VitaLlama_Colab_Server.ipynb`](VitaLlama_Colab_Server.ipynb) to [Google Colab](https://colab.research.google.com/).
2. Select **T4 GPU** Accelerator: `Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ `T4 GPU`.
3. Mount Google Drive containing your fine-tuned weights at `/content/drive/MyDrive/Nutritionist_LLM_Phase1/merged_model_drive` (or let it fall back to base `unsloth/Meta-Llama-3.1-8B-bnb-4bit`).

### Step 2: Execute Notebook Cells
1. Run **Cells 1 to 5** to load dependencies, weights, and initialize the memory engine.
2. In **Cell 6**, optionally add your Ngrok auth token and execute to launch the server.
3. *(Alternative)* If you do not have an Ngrok token, run **Cell 7** to start a zero-setup Cloudflare tunnel.
4. Copy the public URL printed at the bottom of the cell:
   ```text
   👉 Public URL: https://xxxx-xx-xx.ngrok-free.app
   ```

### Step 3: Launch Frontend
1. Open [`index.html`](index.html) in your browser (or use VS Code Live Server).
2. Paste your public URL into the **Colab API URL** bar at the top.
3. Click **Connect** (the status dot turns green).
4. Start consulting with VitaLlama!

---

## 📂 Repository Structure

```
├── Chatbot_fine_tuning.ipynb      # Phase 1: QLoRA SFT training & weight merging pipeline
├── VitaLlama_Colab_Server.ipynb   # Phase 2: FastAPI + Unsloth + 3-Tier Memory + Tunnel host
├── index.html                     # Frontend: Semantic HTML layout & structural markup
├── style.css                      # Frontend: Theme tokens, responsive layout, glassmorphic UI
├── app.js                         # Frontend: SSE stream parser & memory dashboard logic
└── README.md                      # Complete system documentation & architecture guide
```

---

## ⚠️ Clinical Safety & Disclaimer

VitaLlama is trained for educational and informational dietary guidance. It is **not** a certified medical diagnostic tool. In compliance with clinical AI guardrails:
- The system prompt explicitly commands the model to **never diagnose severe diseases** (such as Chronic Kidney Disease or Type 1 Diabetes).
- It instructs patients exhibiting red-flag symptoms or high-risk metabolic issues to seek immediate consultation with a licensed physician or registered dietitian.

