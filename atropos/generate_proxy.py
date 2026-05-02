"""CPU-only HTTP proxy between atropos environments and verl's internal vLLM.

FIXES:
- Fast request admission: lock held for microseconds, no condition variables
- Massively parallel requests: eliminates global contention and notify_all storms
- Fast release: lock-free decrementing
- Correct /pause: occasional stale reads handled via polling
"""

import argparse
import asyncio
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

_backend_urls: list[str] = []
_rr_counter: int = 0
_model: str = ""
_client: httpx.AsyncClient | None = None

# -----------------------------------------------------------------------------
# Drain / synchronization state
# -----------------------------------------------------------------------------

_paused: bool = False
_in_flight: int = 0

_admission_lock: asyncio.Lock | None = None
_resume_event: asyncio.Event | None = None

_DRAIN_TIMEOUT: float = 300.0
_GENERATION_TIMEOUT: float = 300.0


def _next_backend() -> str:
    """Round-robin across backend vLLM servers."""
    global _rr_counter

    url = _backend_urls[_rr_counter % len(_backend_urls)]
    _rr_counter += 1
    return url


@asynccontextmanager
async def lifespan(app):
    global _client, _admission_lock, _resume_event

    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(_GENERATION_TIMEOUT, connect=10)
    )

    _admission_lock = asyncio.Lock()
    _resume_event = asyncio.Event()
    _resume_event.set()

    yield

    await _client.aclose()


app = FastAPI(lifespan=lifespan)


# -----------------------------------------------------------------------------
# Request admission helpers
# -----------------------------------------------------------------------------

async def _acquire_generation_slot():
    """
    Fast request admission.
    Acquires slot natively unless paused. If paused, waits on the resume event
    outside of the lock to prevent global contention.
    """
    global _in_flight

    while True:
        async with _admission_lock:
            if not _paused:
                _in_flight += 1
                return

            wait_event = _resume_event

        await wait_event.wait()


async def _release_generation_slot():
    """
    Fast release.
    No locking needed because only /pause cares, occasional stale reads are 
    fine INSIDE the pause loop, and the admission race is already eliminated.
    """
    global _in_flight
    _in_flight -= 1


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------

@app.get("/health")
async def health():
    """
    Passthrough to backend /health.

    Returns 503 while paused.
    """
    if _paused:
        return Response(status_code=503)

    try:
        resp = await _client.get(
            f"{_backend_urls[0]}/health",
            timeout=5,
        )
        return Response(status_code=resp.status_code)

    except Exception:
        return Response(status_code=503)


# -----------------------------------------------------------------------------
# Legacy /generate endpoint
# -----------------------------------------------------------------------------

@app.post("/generate")
async def generate(request: Request):
    """
    Translate atropos /generate -> /v1/completions.
    """
    await _acquire_generation_slot()

    try:
        data = await request.json()

        # pass token IDs directly to vLLM
        prompt = data.get("prompt")

        if isinstance(prompt, dict):
            prompt = prompt["prompt_token_ids"]

        # atropos may send logprobs=0
        # OpenAI API requires >=1
        logprobs_raw = data.get("logprobs")

        logprobs_val = (
            max(1, int(logprobs_raw))
            if logprobs_raw is not None
            else 1
        )

        completions_req = {
            "model": _model,
            "prompt": prompt,
            "n": data.get("n", 1),
            "max_tokens": data.get("max_tokens", 16),
            "temperature": data.get("temperature", 1.0),
            "top_p": data.get("top_p", 1.0),
            "logprobs": logprobs_val,
            "return_tokens_as_token_ids": True,
        }

        if data.get("stop"):
            completions_req["stop"] = data["stop"]

        backend_url = _next_backend()

        try:
            resp = await _client.post(
                f"{backend_url}/v1/completions",
                json=completions_req,
            )

        except Exception as e:
            return JSONResponse(
                {"error": str(e)},
                status_code=503,
            )

        if resp.status_code != 200:
            return Response(
                status_code=resp.status_code,
                content=resp.content,
            )

        result = resp.json()

        # translate response back to atropos format
        texts = []
        logprobs_out = []
        finish_reasons = []

        for choice in result.get("choices", []):
            texts.append(choice["text"])

            finish_reasons.append(
                choice.get("finish_reason") or "length"
            )

            lp = choice.get("logprobs")

            if lp and lp.get("tokens"):
                seq = []

                for tok_str, tok_lp in zip(
                    lp["tokens"],
                    lp["token_logprobs"],
                    strict=True,
                ):
                    tid = int(tok_str.split(":")[1])

                    seq.append([
                        {
                            tid: tok_lp if tok_lp is not None else 0.0
                        }
                    ])

                logprobs_out.append(seq)

            else:
                logprobs_out.append([])

        return JSONResponse(
            {
                "text": texts,
                "logprobs": logprobs_out,
                "finish_reasons": finish_reasons,
            }
        )

    finally:
        await _release_generation_slot()


# -----------------------------------------------------------------------------
# OpenAI completions passthrough
# -----------------------------------------------------------------------------

@app.post("/v1/completions")
async def v1_completions(request: Request):

    await _acquire_generation_slot()

    try:
        data = await request.json()

        # overwrite model name
        data["model"] = _model

        backend_url = _next_backend()

        try:
            resp = await _client.post(
                f"{backend_url}/v1/completions",
                json=data,
            )

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )

        except Exception as e:
            return JSONResponse(
                {"error": str(e)},
                status_code=503,
            )

    finally:
        await _release_generation_slot()


# -----------------------------------------------------------------------------
# OpenAI chat completions passthrough
# -----------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request):

    await _acquire_generation_slot()

    try:
        data = await request.json()

        data["model"] = _model

        backend_url = _next_backend()

        try:
            resp = await _client.post(
                f"{backend_url}/v1/chat/completions",
                json=data,
            )

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )

        except Exception as e:
            return JSONResponse(
                {"error": str(e)},
                status_code=503,
            )

    finally:
        await _release_generation_slot()


# -----------------------------------------------------------------------------
# Pause / Resume
# -----------------------------------------------------------------------------

@app.post("/pause")
async def pause():
    """
    Stop admitting new requests and wait for all in-flight requests to drain.
    Occasionally polling _in_flight works efficiently because new admissions
    are blocked behind _admission_lock and _paused=True.
    """
    global _paused

    async with _admission_lock:
        _paused = True
        _resume_event.clear()

    deadline = asyncio.get_running_loop().time() + _DRAIN_TIMEOUT

    while _in_flight > 0:
        if asyncio.get_running_loop().time() > deadline:
            async with _admission_lock:
                _paused = False
                _resume_event.set()

            return JSONResponse(
                {
                    "status": "timeout",
                    "in_flight": _in_flight,
                },
                status_code=504,
            )

        await asyncio.sleep(0.01)

    return JSONResponse(
        {
            "status": "paused",
            "drained": True,
        }
    )


@app.post("/resume")
async def resume():
    """Resume request admission."""
    global _paused

    async with _admission_lock:
        _paused = False
        _resume_event.set()

    return JSONResponse({"status": "resumed"})


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="atropos /generate -> vLLM proxy"
    )

    parser.add_argument(
        "--backend-url",
        required=True,
        help=(
            "comma-separated backend URLs "
            "(e.g. http://ip:8000,http://ip:8001)"
        ),
    )

    parser.add_argument(
        "--model",
        required=True,
        help="served model name",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9004,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
    )

    parser.add_argument(
        "--drain-timeout",
        type=float,
        default=300.0,
    )

    parser.add_argument(
        "--generation-timeout",
        type=float,
        default=300.0,
    )

    args = parser.parse_args()

    global _backend_urls
    global _model
    global _DRAIN_TIMEOUT
    global _GENERATION_TIMEOUT

    _backend_urls = [
        url.rstrip("/")
        for url in args.backend_url.split(",")
    ]

    _model = args.model
    _DRAIN_TIMEOUT = args.drain_timeout
    _GENERATION_TIMEOUT = args.generation_timeout

    print(
        f"generate proxy: {args.host}:{args.port} "
        f"-> {_backend_urls} "
        f"(model={_model})"
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
