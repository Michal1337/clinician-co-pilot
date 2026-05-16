from typing import Any, Dict, List, Literal, TypedDict

import numpy as np
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from models import PIPE, PIPE_ASR
from prompts import PROMPT_SUMMARIZE_TRANSCRIPTION
from utils import parse_response_json

SAMPLE_RATE = 16000
CHUNK_SECONDS = 10
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
OVERLAP_RATIO = 0.1
STEP_SAMPLES = int(CHUNK_SAMPLES * (1 - OVERLAP_RATIO))

# Gating on the structured-summary LLM pass — every Nth chunk OR a big-enough delta.
SUMMARIZE_EVERY_N_CHUNKS = 3
SUMMARIZE_MIN_DELTA_CHARS = 200


class AgentState(TypedDict):
    subject_id: int
    action: Literal["search_text", "search_imaging", "finish"]
    query: str
    allowed_years: int
    retrieved_docs: List[List[Document]]
    retrieved_docs_str: str
    num_retriev_text: int
    num_retriev_img: int

    summary: Dict[str, Any]
    action_history: List[Dict[str, Any]]
    step: int

    question: str
    chat_history: List[str]

    audio_chunk: np.ndarray
    transcript_chunk: str
    full_transcript: str
    conversation_summary: Any
    chunk_count: int
    last_summary_at_len: int


def _stitch_chunk(full: str, chunk: str, max_overlap_words: int = 8) -> str:
    """Append a fresh ASR chunk to the running transcript, removing any
    word-level overlap (chunks share ~10% audio so the last few words of
    `full` and the first few of `chunk` often duplicate)."""
    chunk = (chunk or "").strip()
    if not chunk:
        return full
    if not full:
        return chunk
    tail = full.split()[-max_overlap_words:]
    head = chunk.split()
    overlap = 0
    for k in range(min(len(tail), len(head)), 0, -1):
        if [w.lower() for w in tail[-k:]] == [w.lower() for w in head[:k]]:
            overlap = k
            break
    deduped = " ".join(head[overlap:])
    if not deduped:
        return full
    sep = "" if full.endswith((" ", "\n")) else " "
    return full + sep + deduped


def node_transcribe(state: AgentState):
    waveform = state["audio_chunk"]
    result = PIPE_ASR(waveform, sampling_rate=SAMPLE_RATE)
    chunk_text = result["text"] or ""

    updated_transcript = _stitch_chunk(state["full_transcript"], chunk_text)

    return {
        "transcript_chunk": chunk_text,
        "full_transcript": updated_transcript,
        "chunk_count": state.get("chunk_count", 0) + 1,
    }


def _should_summarize(state: AgentState) -> str:
    n = state.get("chunk_count", 0)
    if n <= 1:
        return "summarize"
    transcript = state.get("full_transcript", "") or ""
    last_len = state.get("last_summary_at_len", 0)
    grew_enough = (len(transcript) - last_len) >= SUMMARIZE_MIN_DELTA_CHARS
    interval = (n % SUMMARIZE_EVERY_N_CHUNKS) == 0
    return "summarize" if (grew_enough or interval) else "skip"


def node_summarize(state: AgentState):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT_SUMMARIZE_TRANSCRIPTION.format(
                        full_text=state["full_transcript"],
                        conversation_summary=state["conversation_summary"],
                    ),
                }
            ],
        },
    ]

    response = PIPE(messages, max_new_tokens=4000)
    summary = parse_response_json(response)
    if not isinstance(summary, dict) or not summary:
        summary = state["conversation_summary"]
    return {
        "conversation_summary": summary,
        "last_summary_at_len": len(state.get("full_transcript", "")),
    }


graph = StateGraph(AgentState)
graph.add_node("transcribe", node_transcribe)
graph.add_node("summarize", node_summarize)

graph.add_edge(START, "transcribe")
graph.add_conditional_edges(
    "transcribe", _should_summarize, {"summarize": "summarize", "skip": END}
)
graph.add_edge("summarize", END)

AUDIO_AGENT = graph.compile()
