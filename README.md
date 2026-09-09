# PDFQUERY 
A full stack RAG platform with two frontends (React + Streamlit), a FastAPI backend, hybrid retrieval with cross encoder
reranking, RAGAS based evaluation, full Dockerization and CI/CD.

🔗 **Live demo (Streamlit):** https://pdfquery-crebnhvbmtkvgtfx4alwuf.streamlit.app/
🔗 **Live React UI:** lighthearted-bienenstitch-7ae9c9.netlify.app *(free-tier backend — use small PDFs like one page or two page)*
🔗 **Live API docs (Swagger):** https://a56228-09a1.a.jrnm.app/docs

## Features
- **PDF Q&A with citations** — every answer ships with its source passages
- **Hybrid retrieval** — dense (FastEmbed + Chroma) + sparse (BM25), fused with
  Reciprocal Rank Fusion, then **FlashRank cross-encoder reranking**
- **Persistent conversations** — SQLite-backed history, shared schema across
  both frontends; reopen, continue, and delete old chats
- **Re-chat with old PDFs** — uploads are stored on disk and re-indexed on demand
- **Focused answering** — prompt-engineered to answer *only what was asked*
  (validated with RAGAS `answer_relevancy`)
- **Two production UIs** — ChatGPT-style React app and a polished Streamlit app,
  both driven by the *same* RAG engine
- **Evaluation harness** — auto-generated Q/A datasets scored with RAGAS
  (faithfulness, answer relevancy, context precision/recall)
- **Containerized & automated** — multi-stage Docker builds, one-command full
  stack, GitHub Actions CI/CD publishing images to Docker Hub on every push

##  Architecture

```
┌───────────────┐      ┌───────────────┐
│  React (Vite) │      │   Streamlit   │
│  static build │      │   (in-proc)   │
└──────┬────────┘      └──────┬────────┘
       │ HTTPS/REST           │
┌──────▼──────────────────────▼────────┐
│         FastAPI · server.py          │
│     SQLite conversations + sources   │
└──────┬───────────────────────────────┘
       │
┌──────▼───────────────────────────────┐
│        rag_pipeline.py (shared)      │
│  pypdf → recursive splitter          │
│  FastEmbed(ONNX) + Chroma  ─┐        │
│  BM25 ─────────────────────┼─► RRF   │
│                            ▼         │
│                 FlashRank rerank(4)  │
│                            ▼         │
│            Qwen (via Groq) + strip-  │
│            think post-processing     │
└──────────────────────────────────────┘
```
## Evaluation (RAGAS)

```bash
python generate_eval_set.py     # builds a Q/A eval set from your PDFs
python eval_harness.py          # scores the live pipeline with RAGAS
```

Metrics: **faithfulness · answer_relevancy · context_precision · context_recall**.
Used to validate prompt and retrieval changes (e.g. the focused-answer rule
raised `answer_relevancy` measurably).

---

##  Quickstart (local)

**Prereqs:** Python 3.11, Node 20+, Docker · `.env` with `GROQ_API_KEY=...`

```bash
pip install -r requirements.txt
```

**Option A — Streamlit app**

```bash
streamlit run pdfquery.py
```

**Option B — React + FastAPI**

```bash
uvicorn server:app --reload --port 8000        # backend
cd pdfquery-frontend && npm install && npm run dev   # frontend → :5173
```

**Option C — everything in Docker (one command)**

```bash
docker compose up --build
# frontend → http://localhost:8080 · backend → http://localhost:8000/docs
```

---

##  Docker & CI/CD

- `Dockerfile` (backend): multi-stage, ONNX models **baked into the image** at
  build time → no cold downloads at runtime
- `pdfquery-frontend/Dockerfile`: Node build → nginx serving the static bundle,
  `VITE_API_URL` injected as build arg
- `.github/workflows/deploy.yml`: on every push to `main`, GitHub Actions builds
  both images and publishes them to **Docker Hub** using repository secrets
  (`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`)

---

##  Deployment & free-tier notes

1. FastAPI backend | JustRunMy.App (container) | Free tier = 150 MB RAM → ONNX indexing OOMs on large PDFs; live demo uses small docs |
2. Streamlit demo | Streamlit Community Cloud | Full experience; servers have real resources |
3. React frontend | Netlify (static) | Built with `VITE_API_URL` pointing at the live backend |

> **Engineering note:** free-tier container hosts cap RAM at ~150 MB, which is
> below the working set of an ONNX-based RAG stack. The limitation was
> benchmarked and documented per host (OOM exit-137 crash loops); production
> path is `docker compose` on any ≥1 GB host (Render/AWS/OCI).

## Author

**Abdul Ahad** 
An end to end portfolio project: RAG engineering → evaluation → containerization → CI/CD → deployment → cost/resource benchmarking.
