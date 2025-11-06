# pdfscribe2ds-fastapi

> (WIP) A web service that hosts a tool to convert PDF documents to Markdown format using AI-powered OCR and optional image captioning, shipped via FastAPI.


Forked from [`KnightChaser/pdfscribe2ds(66e9d4)`](https://github.com/KnightChaser/pdfscribe2ds/tree/66e9d45afb4987110b3ec915876efbe3dbdb1922).

## Installation

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
uv pip install timm
```