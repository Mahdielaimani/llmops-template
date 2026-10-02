# Environment Audit — Phase 0

**Audit date:** 2026-09-14
**Machine:** personal laptop (Windows 11 + Docker Desktop / WSL2 backend)
**Status label:** LOCAL / DEMO

This document records what the machine can and cannot do, and which
execution mode the laboratory will use. Every later phase must respect
the resource budget defined here.

---

## 1. Summary

| Area | Result | Verdict |
| --- | --- | --- |
| OS | Windows 11 Pro 10.0.26200, x64 | OK |
| CPU | AMD Ryzen 9 7845HX, 12 cores / 24 threads, 3.0 GHz base | Strong |
| RAM | **15.2 GB total**, 1.9 GB free at audit time | **PRIMARY CONSTRAINT** |
| Disk | C: 953 GB, **33 GB free** | **TIGHT** — Docker holds ~53 GB, ~44 GB reclaimable |
| GPU | NVIDIA GeForce RTX 4070 Laptop, **8 GB VRAM**, compute capability 8.9 (Ada) | Usable, small |
| NVIDIA driver | 581.42 | OK |
| CUDA toolkit (host) | 12.8 (`nvcc` V12.8.61), `CUDA_PATH` set | OK |
| GPU in Docker | `docker run --gpus all … nvidia-smi` → RTX 4070 visible | **WORKS** |
| Docker | 29.2.0, Docker Desktop, WSL2 backend, `nvidia` runtime registered | OK |
| Docker Compose | v5.0.2 | OK |
| Python | 3.14.0 (default), 3.13, **3.12.10**, 3.11.9, miniconda | Use **3.12** |
| uv | 0.8.22 | OK |
| pip | 26.1.2 | OK |
| Git | 2.53.0 | OK |
| kubectl | v1.35.1 | OK |
| Helm | v4.1.0 | OK |
| minikube | v1.37.0, **no cluster running** | Available |
| kind | not installed | Optional |
| Kubernetes contexts | `aks-llmops` (Azure, pre-existing), `minikube` | Cloud ctx exists |
| make | not installed | Need alternative |
| Node | v24.12.0 | OK (Promptfoo) |
| WSL | v2, distro `docker-desktop` only | OK |
| Internet | **offline at audit time** (DNS resolution failed) | Blocks image pulls |

---

## 2. Hardware detail

### CPU
- AMD Ryzen 9 7845HX (Zen 4), 12 physical cores, 24 logical threads.
- Docker Desktop VM sees all 24 CPUs.
- Good for: BM25, reranker (CPU cross-encoder), embedding on CPU, Locust
  load generation, several API replicas.

### RAM — the real bottleneck
- 15.2 GB physical.
- Docker Desktop VM reports `MemTotal = 12.5 GB` (11.7 GiB) — it can
  claim most of host RAM under load.
- Only 1.9 GB free at audit time (Docker Desktop was starting + other
  apps). Expect memory pressure when running:
  - vLLM container (model weights staged through CPU RAM before GPU)
  - Langfuse v3 (needs ClickHouse + MinIO + Redis + Postgres ≈ 3 GB)
  - Qdrant + PostgreSQL + Redis + Prometheus + Grafana + Jaeger
  - N API replicas + workers
- **Consequence:** the stack must be split into Compose profiles and
  never run fully at once. See resource budget in section 6.

### Disk
- 33 GB free on C:.
- `docker system df`:

  | Type | Total | Size | Reclaimable |
| --- | --- | --- | --- |
  | Images | 17 | 31.2 GB | 27.4 GB |
  | Containers | 13 | 209 MB | 209 MB |
  | Volumes | 7 | 84 MB | 0 |
  | Build cache | 87 | 22.0 GB | 16.3 GB |

- A 7B INT4 model ≈ 4–5 GB, a 3B FP16 ≈ 6 GB, embedding + reranker
  models ≈ 1–2 GB, vLLM image ≈ 10+ GB. **Disk cleanup is required
  before Phase 6 (LLM serving).**

### GPU
- NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB VRAM, 8 MiB in use.
- Compute capability 8.9 → supports FP16, BF16, INT8, FP8 tensor cores,
  FlashAttention-2, AWQ/GPTQ kernels.
- Also present: AMD Radeon 610M (iGPU) — ignored.
- Verified from inside a container:

  ```text
  $ docker run --rm --gpus all python:3.11-slim nvidia-smi --query-gpu=name,memory.total --format=csv
  NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB
  ```

  → NVIDIA Container Toolkit is functional through Docker Desktop's
  WSL2 backend. No extra install needed.

#### VRAM budget (8 GB) — what fits

| Model | Precision | Weights | KV cache headroom | Fit? |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B / 1.5B | FP16 | 1–3 GB | 4–6 GB | Yes, comfortable |
| Qwen2.5-3B / Llama-3.2-3B | FP16 | ~6 GB | ~1 GB | Marginal — short context only |
| Qwen2.5-7B / Llama-3.1-8B | INT4 (AWQ/GPTQ) | ~4.5 GB | ~2.5 GB | Yes, limited concurrency |
| 7B / 8B | FP16 | ~15 GB | — | **No** |

The 8 GB ceiling is a feature for this lab: it makes KV-cache pressure,
VRAM OOM, and "GPU is the bottleneck" scenarios easy to reproduce with
real hardware instead of simulation.

---

## 3. Software detail

### Python
- `python` on PATH = 3.14.0. **Do not use.** PyTorch/vLLM/many ML wheels
  lag behind the newest CPython.
- Project will pin **Python 3.12** via `uv` (`.python-version`).
- No `torch`, `vllm`, `transformers`, `fastapi` etc. currently installed
  in 3.12 — clean slate.

### Docker
- Engine 29.2.0, Compose v5.0.2, overlayfs, runtimes: `runc`, `nvidia`.
- Docker Desktop was **stopped** at audit start; started manually.
- Pre-existing networks: `bridge`, `host`, `none`, `business_default`,
  `claim-approval-agent_default`, `llmops-sim_default`.
- Pre-existing containers (other projects), auto-restarted with daemon:
  - `claim-approval-api` → **binds host port 8000**
  - `bizos-db` (postgres:17) → binds host port 5433
  - `llmops-sim-*` (litellm, langfuse v3, rag-api, redis, postgres,
    clickhouse, minio) → all Exited, 6 weeks old
- Pre-existing **cached images** reusable offline:
  `qdrant/qdrant:v1.12.4`, `redis:7-alpine`, `postgres:16-alpine`,
  `postgres:17-alpine`, `prom/prometheus:latest`, `grafana/grafana:latest`,
  `jaegertracing/all-in-one:1.62.0`, `langfuse/langfuse:3`,
  `langfuse/langfuse-worker:3`, `ghcr.io/berriai/litellm:main-stable`,
  `minio/minio:latest`, `clickhouse/clickhouse-server:25.12`,
  `python:3.11-slim`, `hashicorp/terraform:1.9`.

### Kubernetes
- `kubectl` 1.35.1, `helm` 4.1.0, `minikube` 1.37.0 installed.
- No local cluster running. `minikube status` → container does not exist.
- Docker Desktop's built-in Kubernetes is **disabled**.
- Context `aks-llmops` points at an Azure AKS cluster (pre-existing).
  This lab is local-first; AKS is out of scope unless you explicitly opt in.
- Kubernetes phase (32) will use minikube with the docker driver and
  `--gpus all` (minikube ≥1.32 supports NVIDIA GPU passthrough on docker
  driver).

### Missing / not installed
- `make` — the Makefile convention from the master prompt needs a
  substitute. Options: install GNU Make via `winget install GnuWin32.Make`
  or `choco install make`, **or** use `uv run` task scripts / a `justfile`.
  Decision deferred to Phase 1 (your call — see section 7).
- `kind` — optional; minikube suffices.

### Network
- At audit time neither the host nor Docker could resolve
  `registry-1.docker.io`. All image pulls, model downloads (Hugging
  Face) and `uv sync` will fail until connectivity returns.

---

## 4. Ports

Checked host listeners for the ports this lab will use.

| Port | Intended use | Status |
| --- | --- | --- |
| 8000 | API (FastAPI) | **IN USE** by `claim-approval-api` |
| 8080 | API Gateway / alternate API | free |
| 8001 | API replica / model gateway | free |
| 5432 | PostgreSQL | free (5433 taken by `bizos-db`) |
| 6379 | Redis | free |
| 6333 | Qdrant HTTP | free |
| 5000 | MLflow | free |
| 9090 | Prometheus | free |
| 3000 | Grafana / Langfuse | free |
| 4317 / 4318 | OTLP gRPC / HTTP | free |
| 16686 | Jaeger UI | free |
| 9100 | node-exporter | free |
| 11434 | Ollama (if used) | free |
| 80 / 443 | Ingress | free |

Plan: the lab's own API will bind **8080** (gateway) and internal
replicas will not publish host ports at all — they are reached through
the load balancer. This sidesteps the 8000 conflict without touching
your other project.

---

## 5. Environment variables

Only names are recorded; values were never printed.

| Variable | Present | Note |
| --- | --- | --- |
| `CUDA_PATH`, `CUDA_PATH_V12_8` | yes | host CUDA 12.8 |
| `OPENAI_API_KEY` | no | needed for MODE A (external API) |
| `ANTHROPIC_API_KEY` | no | alternative for MODE A |
| `HF_TOKEN` | no | needed only for gated HF models |
| `AZURE_*` | no | AKS context exists but no creds in env |

No LLM provider credentials exist in the environment. MODE A will
require you to supply one (see section 7).

---

## 6. Execution mode decision

Three modes were evaluated:

| Mode | Feasible? | Reason |
| --- | --- | --- |
| LOCAL CPU MODE | Yes | 24 threads; small models via llama.cpp/Ollama or mock |
| LOCAL GPU MODE | Yes, constrained | 8 GB VRAM, GPU passthrough verified; ≤7B INT4 |
| HYBRID MODE | **Yes — SELECTED** | GPU for inference lessons, external API optional, CPU/mock fallback |

### Decision: HYBRID MODE

- **MODE B (local open-source LLM)** is the primary path for inference
  phases (4–7, 30–31, 37). vLLM runs **inside Docker** (vLLM has no
  native Windows build; the WSL2 backend gives it a Linux kernel + CUDA).
  Model class: Qwen2.5-1.5B-Instruct FP16 for fast iteration,
  Qwen2.5-7B-Instruct-AWQ for realism. Ollama or llama.cpp as a lighter
  fallback if vLLM's memory footprint is too large for 15 GB RAM.
- **MODE A (external API)** is wired in from Phase 2 as a provider behind
  the same interface, but only activated when you supply a key. Used for
  LLM-as-a-Judge (Phase 12) where a stronger model is valuable.
- **MODE C (mock)** is a deterministic fake provider used by unit tests,
  CI, and any phase where the GPU is busy or unavailable.

### Resource budget per Compose profile

Never run all profiles together. Approximate RAM in the Docker VM:

| Profile | Services | RAM est. |
| --- | --- | --- |
| `core` | api ×1, postgres, redis, qdrant | ~1.5 GB |
| `inference` | vllm (7B AWQ) | ~3–4 GB RAM + ~7 GB VRAM |
| `observability` | prometheus, grafana, jaeger/otel-collector | ~1 GB |
| `tracing-llm` | Langfuse v3 (web, worker, clickhouse, minio, redis, pg) | ~3 GB — **evaluate lighter alternative (Phoenix) in Phase 22** |
| `eval` | mlflow | ~0.5 GB |
| `scale` | api ×4, worker ×2 | ~1.5 GB |
| `edge` | Kong OSS DB-less (gateway + LB) | ~0.3 GB |
| `load` | locust | ~0.5 GB |

Typical combos: `core+inference` (≈5.5 GB), `core+inference+observability`
(≈6.5 GB), `core+scale+observability+load` (≈5 GB, mock/API provider), `core+edge+scale+load` (≈4 GB).
All within ~11.7 GB VM ceiling with margin.

---

## 7. Actions required from you before Phase 1

Nothing blocks Phase 1 (repo foundation) except network. The rest are
needed by later phases; listed now so you can plan.

### 7.1 Network (blocks Phase 1 `uv sync` and any `docker pull`)
- **WHAT:** restore internet access.
- **VERIFY:** `curl -I https://pypi.org` returns `HTTP/2 200`, and
  `docker pull hello-world` succeeds.

### 7.2 Task runner (Phase 1 decision)
- **WHAT:** choose `make` vs `just` vs plain `uv run` scripts.
- **WHY:** `make` is not installed; Makefile is the master-prompt default.
- **HOW (if make):** `winget install GnuWin32.Make` then reopen terminal.
- **VERIFY:** `make --version`.
- Default if you do not answer: `justfile` + `uv` (cross-platform, no
  install friction; I will document the mapping to Makefile targets).

### 7.3 Disk cleanup (needed before Phase 6 — LLM serving)
- **WHAT:** reclaim Docker space. ~44 GB reclaimable.
- **WHY:** 33 GB free is not enough for vLLM image (~10 GB) + models (~5–10 GB) + new service images.
- **HOW:** review first with `docker system df -v`, then
  `docker builder prune -a` (16 GB, safe) and `docker image prune -a`
  (removes images not used by any container — **this includes your
  cached qdrant/grafana/langfuse images**; prefer targeted `docker rmi`).
- **I will not run prune myself** — it touches your other projects.
- **VERIFY:** `docker system df` and free disk ≥ 60 GB.

### 7.4 External LLM API key (needed for MODE A, first used in Phase 2 optionally, required in Phase 12 LLM-as-a-Judge)
- **WHAT:** an `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
- **WHERE:** `.env` file at repo root (gitignored); template in `.env.example`.
- **VERIFY:** Phase 2 `/chat` request with `provider=openai` returns a real answer.
- Not required to start. Phase 2 runs on MODE C (mock) and MODE B.

### 7.5 Hugging Face token (only if a gated model is chosen, Phase 6)
- Qwen2.5 models are ungated. Llama-3.x models are gated. Default plan
  uses Qwen → no token needed.

### 7.6 Port 8000 (no action needed)
- Lab uses 8080 for the gateway. If you later want 8000, stop
  `claim-approval-api` yourself: `docker stop claim-approval-api`.

### 7.7 Memory (advisory)
- Close heavy apps during inference/load-test phases. Consider a
  `%USERPROFILE%\.wslconfig` with `memory=10GB` to cap the Docker VM so
  Windows itself keeps ≥5 GB. **Your decision; I will not edit it.**

---

## 8. How this audit was performed

Commands used (all read-only except starting Docker Desktop):

```text
Get-CimInstance Win32_OperatingSystem / Win32_Processor / Win32_ComputerSystem / Win32_LogicalDisk / Win32_VideoController
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,compute_cap --format=csv
nvcc --version
py -0 ; py -3.12 --version ; py -3.11 --version
uv --version ; pip --version ; git --version ; node --version
docker --version ; docker compose version ; docker info ; docker network ls ; docker ps -a ; docker images ; docker system df
docker run --rm --gpus all python:3.11-slim nvidia-smi
kubectl version --client ; kubectl config get-contexts ; helm version ; minikube status
Get-NetTCPConnection -State Listen
Get-ChildItem env: | Where-Object Name -match "<secret-ish patterns>"   # names + length only
```

Secrets policy: environment variable **values were never displayed**;
only the variable name and character count were inspected.
