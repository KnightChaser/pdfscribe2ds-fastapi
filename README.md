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

## Usage

Utilize `uv run` command to execute the FastAPI server. It will automatically load two necessary AI models (`DeepSeek-OCR` and `DeepSeek-VL2-tiny`) on each designated GPU devices. `--workers 1` option is mandatory because the AI models cannot be shared across multiple worker processes, and GPU resources must be managed globally.

```sh
uv run -- uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

When the FastAPI service is up, you can try the functionality of the service by sending an example PDF document you want to be processed automatically with `curl` command as follows:

```sh
curl -f -X POST \
  -F "file=@./example/investment_report.pdf;type=application/pdf" \
  "http://localhost:8000/v1/process/pdf?rewrite_mode=append&" \
  -o out.zip
```

