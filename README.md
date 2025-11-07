# pdfscribe2ds-fastapi

> A web service that hosts a tool to convert PDF documents to Markdown format using AI-powered OCR and optional image captioning, shipped via FastAPI.

Project has diverged and further developed from [`KnightChaser/pdfscribe2ds(66e9d4)`](https://github.com/KnightChaser/pdfscribe2ds/tree/66e9d45afb4987110b3ec915876efbe3dbdb1922).

## Overview

This project provides a FastAPI-based web service for converting PDF documents into structured Markdown format. It uses AI models for optical character recognition (OCR) and optional image captioning to extract and enhance content from PDFs. The service processes PDFs page by page, generating Markdown files and preserving images with captions.

Key features include:
- PDF to Markdown conversion using DeepSeek-OCR
- Optional image captioning with DeepSeek-VL2
- GPU-accelerated processing
- Single-job admission control to manage GPU resources
- Returns results as a ZIP archive containing Markdown files and images

## Prerequisites

- Python 3.12 or higher
- GPU with sufficient memory (>= 80 GiB recommended in compound, both for `DeepSeek/DeepSeek-OCR` and `DeepSeek/DeepSeek-VL2-tiny`)
- `uv`, the Python3 package manager

## Installation

1. Create a virtual environment:
   ```bash
   uv venv
   ```

2. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```

4. Install additional packages for AI models:
   ```bash
   uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
   uv pip install timm
   ```

## Configuration

The service can be configured using environment variables. Key settings include:

- `MODEL_OCR`: OCR model name (default: deepseek-ai/DeepSeek-OCR)
- `MODEL_VL2`: Vision-language model name (default: deepseek-ai/deepseek-vl2-tiny)
- `GPU_MEM_OCR`: GPU memory fraction for OCR model (default: 0.70)
- `GPU_MEM_VL2`: GPU memory fraction for VL2 model (default: 0.70)
- `OCR_DEVICE`: GPU device for OCR (default: "0")
- `VL2_DEVICE`: GPU device for VL2 (default: "1")
- `GPU_SLOTS`: Number of concurrent GPU jobs (default: 1)

Set these variables before running the service if you need to customize the configuration.

## Usage

### Starting the Service

Run the FastAPI server using uv:

```bash
uv run -- uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Note: The `--workers 1` option is required because AI models cannot be shared across multiple worker processes, and GPU resources must be managed globally.

The service will load the necessary AI models (DeepSeek-OCR and DeepSeek-VL2-tiny) on startup.

### Processing a PDF

Once the service is running, you can process a PDF by sending a POST request to the `/v1/process/pdf` endpoint. Here's an example using curl:

```bash
curl -f -X POST \
  -F "file=@./example/investment_report.pdf;type=application/pdf" \
  "http://localhost:8000/v1/process/pdf?rewrite_mode=append" \
  -o out.zip
```

This will process the PDF and return a ZIP file containing the Markdown output and any extracted images.

### API Endpoints

- `GET /v1/health`: Check service health and loaded models
- `GET /v1/models/status`: Get current model status and busy state
- `POST /v1/process/pdf`: Process a PDF file and return results as ZIP

For detailed API documentation, visit `/docs` or `/redoc` when the service is running.

### Parameters

The `/v1/process/pdf` endpoint accepts the following parameters:

- `file`: The PDF file to process (required)
- `dpi`: Resolution for PDF rendering (default: 200)
- `rewrite_mode`: Caption mode - "append" or "replace" (default: "append")
- `seed`: Random seed for processing (optional)
- `wait_if_busy`: Wait for GPU availability if busy (default: false)
- `timeout_s`: Maximum wait time in seconds if busy (default: 0.0)

## Contribution

Contribution in any forms is welcomed! >_<