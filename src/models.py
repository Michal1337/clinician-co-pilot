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

# Sampling: deterministic by default. Greedy decoding (T=0) + fixed seed
# means the planner, summary, alerts, SOAP draft, and chat are all
# reproducible — same inputs → same outputs. Critical for re-recording
# the demo video and debugging the agent.
LLM_TEMPERATURE = float(os.environ.get("CLINICIAN_LLM_TEMPERATURE", "0.0"))
LLM_SEED = int(os.environ.get("CLINICIAN_LLM_SEED", "0"))


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
        has_image = False
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
                    has_image = True
        # Unwrap text-only content to a plain string. Some OpenAI-compatible
        # servers (including certain vLLM builds) are strict about the
        # list-of-parts shape and only accept it for genuinely multimodal
        # messages; passing list-content for a pure-text turn can fail in
        # subtle ways (e.g. response coming back as a raw string).
        if not has_image:
            merged = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
            out.append({"role": role, "content": merged})
        else:
            if not parts:
                parts = [{"type": "text", "text": ""}]
            out.append({"role": role, "content": parts})
    return out


def _extract_completion_text(resp: Any) -> str:
    """Pull assistant text from an OpenAI-style ChatCompletion. Surfaces
    a clear error if the shape is unexpected — some vLLM error paths
    return a raw string and the old ``resp.choices[0]`` crash hides
    what's actually being said."""
    if isinstance(resp, str):
        raise RuntimeError(
            "vLLM returned a string instead of a ChatCompletion. "
            f"First 500 chars: {resp[:500]!r}"
        )
    choices = getattr(resp, "choices", None)
    if not choices:
        return ""
    try:
        return choices[0].message.content or ""
    except (AttributeError, IndexError) as e:
        raise RuntimeError(
            f"Unexpected vLLM response shape ({type(resp).__name__}): {resp!r}"
        ) from e


class _VLLMPipe:
    """Drop-in for the HF `pipeline("image-text-to-text")` shape.

    Returns ``[{"generated_text": <messages + assistant reply>}]`` so the
    existing ``parse_response_json`` / ``extract_assistant_text`` helpers
    in `utils.py` keep working without modification.

    All calls use ``temperature=LLM_TEMPERATURE`` (default 0) and a fixed
    ``seed`` so agent behaviour is deterministic. Any ``do_sample`` /
    ``temperature`` kwargs passed by older call sites are swallowed by
    ``**_`` and ignored — the policy lives in one place.
    """

    model_name = LLM_SERVED_NAME

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 2000,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        oai_messages = _to_oai_messages(messages)
        resp = _client.chat.completions.create(
            model=self.model_name,
            messages=oai_messages,
            max_tokens=max_new_tokens,
            temperature=LLM_TEMPERATURE,
            seed=LLM_SEED,
        )
        text = _extract_completion_text(resp)
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
        **_: Any,
    ) -> Iterator[str]:
        """Yield assistant-content deltas via OpenAI streaming."""
        oai_messages = _to_oai_messages(messages)
        stream = _client.chat.completions.create(
            model=self.model_name,
            messages=oai_messages,
            max_tokens=max_new_tokens,
            temperature=LLM_TEMPERATURE,
            seed=LLM_SEED,
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
