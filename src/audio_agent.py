import warnings
from typing import Any, Dict, List, Literal, TypedDict

import numpy as np
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from models import PIPE, PIPE_ASR
from prompts import PROMPT_SUMMARIZE_TRANSCRIPTION
from utils import extract_response_json, normalize_medasr

warnings.filterwarnings("ignore")


SAMPLE_RATE = 16000
CHUNK_SECONDS = 10
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
STEP_SAMPLES = CHUNK_SAMPLES - int(SAMPLE_RATE * 0.5) 

class AgentState(TypedDict):
    subject_id: int
    action: Literal["search_text", "search_imaging", "finish"]
    query: str
    allowed_years: int
    retrieved_docs: List[List[Document]]
    retrieved_docs_str: str
    num_retriev_text: int
    num_retriev_img: int

    # Stage 1
    summary: Dict[str, Any]
    action_history: List[Dict[str, Any]]
    step: int

    # Stage 2
    question: str
    chat_history: List[str]
    answer_llm: str

    # Stage 3
    audio_chunk: np.ndarray
    transcript_chunk: str
    full_transcript: List[str]
    conversation_summary: str


def node_transcribe(state: AgentState):
    waveform = state["audio_chunk"]

    result = PIPE_ASR(waveform, sampling_rate=SAMPLE_RATE)
    text = result["text"]

    updated_transcript = normalize_medasr(state["full_transcript"] + text)

    return {
        "transcript_chunk": text,
        "full_transcript": updated_transcript
    }

def node_summarize(state: AgentState):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_SUMMARIZE_TRANSCRIPTION.format(full_text=state["full_transcript"], conversation_summary=state["conversation_summary"])}
            ],
        },
    ]

    response = PIPE(messages, max_new_tokens=4000)
    output = extract_response_json(response[0]["generated_text"][-1]["content"])
    return {"conversation_summary": output}

graph = StateGraph(AgentState)
graph.add_node("transcribe", node_transcribe)
graph.add_node("summarize", node_summarize)

graph.add_edge(START, "transcribe")
graph.add_edge("transcribe", "summarize")
graph.add_edge("summarize", END)

AUDIO_AGENT = graph.compile()

# AUDIO_AGENT.get_graph().draw_mermaid_png(output_file_path="../assets/audio_agent.png")