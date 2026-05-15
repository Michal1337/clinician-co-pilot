"""Model wiring.

The LLM (Gemma 4) is served by **vLLM** behind an OpenAI-compatible HTTP
API for throughput; we adapt it back into a `PIPE(messages, ...)` shape
so every existing call site in `agent_demo.py` / `audio_agent.py` keeps
working without changes. ASR stays on the HuggingFace transformers
pipeline because vLLM does not serve speech models.

Start the vLLM server in a separate shell before launching the app:

    CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-4-26B-A4B-it \\
        --port 8000 --max-model-len 32768 --dtype bfloat16 \\
        --gpu-memory-utilization 0.9

Override the endpoint via ``CLINICIAN_VLLM_URL`` if it lives elsewhere.
"""

import base64
import os
from typing import Any, Dict, Iterator, List

from openai import OpenAI
from transformers import pipeline

from model_paths import resolve_model

LLM_MODEL = 'google/gemma-4-26B-A4B-it'
ASR_MODEL = resolve_model(os.environ.get("CLINICIAN_ASR", "google/medasr"))
ASR_DEVICE = os.environ.get("CLINICIAN_ASR_DEVICE", "cuda:1")

# vLLM endpoint: the app talks to it as if it were OpenAI. Auth is unused
# in the default local-serve mode, but the SDK requires *some* key.
VLLM_URL = os.environ.get("CLINICIAN_VLLM_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("CLINICIAN_VLLM_API_KEY", "EMPTY")
# The "model name" the client sends. vLLM serves whatever you passed to
# `vllm serve`; we default to the same id so they match. If you start
# vllm with `--served-model-name clinician-llm`, set CLINICIAN_LLM_SERVED
# to "clinician-llm" so the client request matches.
LLM_SERVED_NAME = os.environ.get("CLINICIAN_LLM_SERVED", LLM_MODEL)


_client = OpenAI(base_url=VLLM_URL, api_key=VLLM_API_KEY)


def _encode_image_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"


def _to_oai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert HF transformers chat shape to OpenAI chat shape, inlining
    local image paths as data URIs so vLLM can read them without filesystem
    access."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": str(content)})
            continue

        parts: List[Dict[str, Any]] = []
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t == "text":
                parts.append({"type": "text", "text": p.get("text", "")})
            elif t == "image":
                url = p.get("url") or p.get("image") or ""
                if url and not url.startswith(("http://", "https://", "data:")):
                    if os.path.exists(url):
                        url = _encode_image_to_data_uri(url)
                if url:
                    parts.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
        if not parts:
            parts = [{"type": "text", "text": ""}]
        out.append({"role": role, "content": parts})
    return out


def _temperature(do_sample: bool) -> float:
    return 0.7 if do_sample else 0.0


class _VLLMPipe:
    """Drop-in for the HF `pipeline("image-text-to-text")` shape.

    Returns ``[{"generated_text": <messages + assistant reply>}]`` so the
    existing ``parse_response_json`` / ``extract_assistant_text`` helpers
    in `utils.py` keep working without modification.
    """

    model_name = LLM_SERVED_NAME

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 2000,
        do_sample: bool = False,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        oai_messages = _to_oai_messages(messages)
        resp = _client.chat.completions.create(
            model=self.model_name,
            messages=oai_messages,
            max_tokens=max_new_tokens,
            temperature=_temperature(do_sample),
        )
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        return [
            {
                "generated_text": list(messages)
                + [{"role": "assistant", "content": text}]
            }
        ]

    def stream(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 2000,
        do_sample: bool = False,
    ) -> Iterator[str]:
        """Yield assistant-content deltas via OpenAI streaming."""
        oai_messages = _to_oai_messages(messages)
        stream = _client.chat.completions.create(
            model=self.model_name,
            messages=oai_messages,
            max_tokens=max_new_tokens,
            temperature=_temperature(do_sample),
            stream=True,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
            except (IndexError, AttributeError):
                delta = None
            if delta:
                yield delta


PIPE = _VLLMPipe()

PIPE_ASR = pipeline("automatic-speech-recognition", ASR_MODEL, device=ASR_DEVICE)
