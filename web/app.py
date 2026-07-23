import os
import json
import uuid
import httpx
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="GenSyntax Web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
VLLM_URL = os.environ.get("VLLM_API_URL", "http://localhost:8000").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "MoonTideF/Llama-GenSyntax")

TASK_CONFIG = {
    "plasmid_host": {
        "name": "Plasmid Host Identification",
        "prompt_template": 'This is the list of protein products encoded by a plasmid. Which bacterial host is this plasmid most likely to come from? Answer strictly in the following format: [genus, species, strain]\n{input}',
        "has_system": True,
        "max_tokens": 100,
        "placeholder": "Enter plasmid protein products, e.g.:\n[DUF3560 domain-containing protein][hypothetical protein][transposase][replication protein]",
    },
    "gene_function": {
        "name": "Gene Function Prediction",
        "prompt_template": '{input}',
        "has_system": True,
        "max_tokens": 100,
        "placeholder": "Enter gene information for function prediction...",
    },
    "genome_assembly": {
        "name": "Contig Order Prediction",
        "prompt_template": '{input}',
        "has_system": True,
        "max_tokens": 1024,
        "placeholder": "Enter the contig-order prediction prompt...",
    },
    "gene_essentiality": {
        "name": "Gene Essentiality Prediction",
        "prompt_template": '{input}',
        "has_system": True,
        "max_tokens": 1024,
        "placeholder": "Enter gene information for essentiality prediction...",
    },
    "minimal_genome": {
        "name": "Minimal Genome Inference",
        "prompt_template": '{input}',
        "has_system": True,
        "max_tokens": 1024,
        "placeholder": "",
        "input_mode": "file",
    },
}

# In-memory job store
jobs: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/tasks")
async def get_tasks():
    return {
        task_id: {
            "name": cfg["name"],
            "placeholder": cfg.get("placeholder", ""),
            "input_mode": cfg.get("input_mode", "text"),
        }
        for task_id, cfg in TASK_CONFIG.items()
    }


async def call_vllm(prompt: str, max_tokens: int = 100, temperature: float = 0,
                    has_system: bool = False) -> str:
    messages = []
    if has_system:
        messages.append({"role": "system", "content": "You are a helpful assistant."})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": True,
        "top_logprobs": 1,
        "min_p": 0,
    }
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(f"{VLLM_URL}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("message", str(data["error"])))
    content = data["choices"][0]["message"]["content"]
    return content.replace("\n", " ").strip()


# ===== Single Prediction =====

@app.post("/api/predict")
async def predict(request: Request):
    body = await request.json()
    task_id = body.get("task")
    user_input = body.get("input", "").strip()

    if not task_id or task_id not in TASK_CONFIG:
        raise HTTPException(400, f"Invalid task. Choose from: {list(TASK_CONFIG.keys())}")
    if not user_input:
        raise HTTPException(400, "Input cannot be empty.")

    cfg = TASK_CONFIG[task_id]
    prompt = cfg["prompt_template"].format(input=user_input)

    try:
        result = await call_vllm(prompt, cfg["max_tokens"], has_system=cfg["has_system"])
    except httpx.ConnectError:
        raise HTTPException(502, "Cannot connect to vLLM server. Make sure it is running.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"vLLM error: {e.response.text}")
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {str(e)}")

    return {"result": result}


# ===== Single Prediction via File Upload =====

@app.post("/api/predict/upload")
async def predict_upload(task: str, file: UploadFile = File(...)):
    if task not in TASK_CONFIG:
        raise HTTPException(400, f"Invalid task.")
    if TASK_CONFIG[task].get("input_mode") != "file":
        raise HTTPException(400, "This task does not support file upload for single prediction.")

    raw = await file.read()
    try:
        inputs = parse_upload(file.filename, raw, task=task)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")

    if not inputs:
        raise HTTPException(400, "No valid inputs found in file.")

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "running", "total": len(inputs), "done": 0, "results": [], "errors": [], "task": None}
    t = asyncio.create_task(run_batch_job(job_id, task, inputs))
    jobs[job_id]["task"] = t
    return {"job_id": job_id, "total": len(inputs)}


# ===== Batch Prediction (async) =====

def parse_minimal_genome(data: list) -> list[str]:
    """Generate one gene-essentiality instruction per CDS in the genome JSON."""
    inputs = []
    for entry in data:
        source = entry.get("Source", "unknown organism")
        products = entry.get("Protein_products", [])
        all_strs = [f"[{p[2].strip()}]" for p in products]
        for i, p in enumerate(products):
            target = p[2].strip()
            context = "".join(s for j, s in enumerate(all_strs) if j != i)
            instruction = (
                f"The following list presents the protein products encoded by {source} chromosome. "
                f"Please predict whether the gene corresponding to the protein product $${target}$$ "
                f"is essential for this organism? "
                f"Answer strictly in the following format: non-essential or essential\n"
                f"{context}"
            )
            inputs.append(instruction)
    return inputs


def parse_upload(filename: str, raw: bytes, task: str = "") -> list[str]:
    """Parse uploaded JSON file into a list of prompt strings.

    Accepted format: JSON array of objects.
    - Minimal-genome tasks: objects with "Protein_products" field (converted via parse_minimal_genome)
    - All other tasks: objects with "Input" / "input" / "instruction" field (used verbatim as prompt)
    """
    if not filename.lower().endswith(".json"):
        raise ValueError("Only JSON files are supported for batch prediction.")
    data = json.loads(raw)
    if not isinstance(data, list):
        data = [data]
    if task == "minimal_genome" or (data and "Protein_products" in data[0]):
        return parse_minimal_genome(data)
    items = []
    for item in data:
        text = item.get("Input") or item.get("input") or item.get("instruction", "")
        items.append(text.strip())
    return [x for x in items if x]


async def run_batch_job(job_id: str, task: str, raw_inputs: list[str]):
    """Background coroutine: call vLLM for each input (inputs are already complete prompts)."""
    cfg = TASK_CONFIG[task]
    job = jobs[job_id]
    results = [""] * len(raw_inputs)
    errors = []

    for i, prompt in enumerate(raw_inputs):
        if not prompt:
            results[i] = "[ERROR] Empty input"
            errors.append(f"Item {i}: empty input")
        else:
            try:
                results[i] = await call_vllm(prompt, cfg["max_tokens"],
                                             has_system=cfg["has_system"])
            except Exception as e:
                errors.append(f"Item {i}: {str(e)}")
                results[i] = f"[ERROR] {str(e)}"

        job["done"] = i + 1

    job["status"] = "done"
    job["results"] = results
    job["errors"] = errors


@app.post("/api/predict/batch")
async def submit_batch(task: str, file: UploadFile = File(...)):
    if task not in TASK_CONFIG:
        raise HTTPException(400, f"Invalid task. Choose from: {list(TASK_CONFIG.keys())}")
    if not (file.filename or "").lower().endswith(".json"):
        raise HTTPException(400, "Only JSON files are supported for batch prediction.")

    raw = await file.read()
    try:
        inputs = parse_upload(file.filename, raw, task=task)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")

    if not inputs:
        raise HTTPException(400, "No valid inputs found in file.")

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "status": "running",
        "total": len(inputs),
        "done": 0,
        "results": [],
        "errors": [],
        "task": None,
    }

    t = asyncio.create_task(run_batch_job(job_id, task, inputs))
    jobs[job_id]["task"] = t
    return {"job_id": job_id, "total": len(inputs)}


@app.get("/api/predict/batch/{job_id}")
async def batch_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {k: v for k, v in job.items() if k != "task"}


@app.delete("/api/predict/batch/{job_id}")
async def batch_cancel(job_id: str):
    job = jobs.pop(job_id, None)
    if job:
        t = job.get("task")
        if t and not t.done():
            t.cancel()
    return {"ok": True}


app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8101"))
    uvicorn.run(app, host="0.0.0.0", port=port)
