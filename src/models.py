import base64
import os
from typing import Any, Dict, Iterator, List

from openai import OpenAI
from transformers import pipeline

from model_paths import resolve_model

LLM_MODEL = 'google/gemma-4-26B-A4B-it'
ASR_MODEL = resolve_model(os.environ.get("CLINICIAN_ASR", "google/medasr"))
ASR_DEVICE = os.environ.get("CLINICIAN_ASR_DEVICE", "cuda:1")

VLLM_URL = os.environ.get("CLINICIAN_VLLM_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("CLINICIAN_VLLM_API_KEY", "EMPTY")
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
        # Some vLLM builds reject list-content for text-only turns.
        if not has_image:
            merged = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
            out.append({"role": role, "content": merged})
        else:
            if not parts:
                parts = [{"type": "text", "text": ""}]
            out.append({"role": role, "content": parts})
    return out


def _extract_completion_text(resp: Any) -> str:
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


def _temperature(do_sample: bool) -> float:
    return 0.7 if do_sample else 0.0


class _VLLMPipe:
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
        do_sample: bool = False,
    ) -> Iterator[str]:
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
