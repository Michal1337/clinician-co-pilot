import warnings
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Literal, Optional, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from audio_agent import *  # noqa: F401,F403  (re-exports CHUNK_SAMPLES, STEP_SAMPLES, AUDIO_AGENT)
from embeddings import IMAGE_EMBEDDINGS, TEXT_EMBEDDINGS
from models import PIPE
from prompts import (
    PROMPT_ALERTS,
    PROMPT_RAG,
    PROMPT_RAG_RETRIEVAL,
    PROMPT_REASON_AND_PLAN,
    PROMPT_SOAP_NOTE,
    PROMPT_UPDATE_TEMPLATE,
    START_CHAT_PROMPT,
)
from templates import CONVERSATION_SUMMARY_TEMPLATE, SUMMARY_TEMPLATE
from utils import (
    GEMMA4_THINK_SEP,
    add_images_to_user_content,
    apply_patch,
    extract_assistant_text,
    imagedoc2str,
    parse_response_json,
    textdoc2str,
    validate_plan,
    windowed_time_decay,
)

warnings.filterwarnings("ignore")


MAX_STEPS = 5

text_vectorstore = FAISS.load_local(
    "../data/vdbs/text_vdb", TEXT_EMBEDDINGS, allow_dangerous_deserialization=True
)
image_vectorstore = FAISS.load_local(
    "../data/vdbs/img_vdb", IMAGE_EMBEDDINGS, allow_dangerous_deserialization=True
)


class AgentState(TypedDict):
    subject_id: int
    action: Literal["search_text", "search_imaging", "finish"]
    query: str
    allowed_years: int
    retrieved_docs: List[List[Document]]
    retrieved_docs_str: str
    num_retriev_text: int
    num_retriev_img: int

    # Stage 1 — pre-visit summary
    summary: Dict[str, Any]
    action_history: List[Dict[str, Any]]
    step: int

    # Stage 2 — chat
    question: str
    chat_history: List[Dict[str, Any]]
    uploaded_image_path: Optional[str]

    # Stage 3 — live transcription
    audio_chunk: Any
    transcript_chunk: str
    full_transcript: str
    conversation_summary: Any
    chunk_count: int
    last_summary_at_len: int


# --- Stage 1 nodes -------------------------------------------------------

def _summary_is_empty(summary: Dict[str, Any]) -> bool:
    """A summary is 'empty' if every section is an empty list / blank.
    The cold-start prompt regression sometimes makes the planner finish
    on step 1 against this template; we coerce a real first query instead."""
    if not isinstance(summary, dict):
        return True
    for v in summary.values():
        if isinstance(v, list) and len(v) > 0:
            return False
        if isinstance(v, str) and v.strip():
            return False
    return True


_COLD_START_PLAN = {
    "action": "search_text",
    "query": "active problems and current diagnoses",
    "allowed_years": 3,
    "thought": "cold-start: summary is empty, retrieve active problems first",
}


def node_reason_and_plan(state: AgentState) -> Dict:
    if state["step"] >= MAX_STEPS:
        return {
            "action": "finish",
            "query": None,
            "action_history": state["action_history"] + ["finish"],
        }

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT_REASON_AND_PLAN.format(
                        summary=state["summary"], action_history=state["action_history"]
                    ),
                }
            ],
        },
    ]
    response = PIPE(messages, do_sample=False, max_new_tokens=2000)
    plan = validate_plan(parse_response_json(response))

    # Guard: never let the planner finish before doing any retrieval at
    # all. The empty-template cold-start frequently confuses the model
    # into "nothing to retrieve, finish" on step 1.
    if (
        plan.get("action") == "finish"
        and not state["action_history"]
        and _summary_is_empty(state["summary"])
    ):
        plan = dict(_COLD_START_PLAN)

    return {
        "action": plan["action"],
        "query": plan.get("query"),
        "allowed_years": plan.get("allowed_years"),
        "action_history": state["action_history"] + [plan],
        "step": state["step"] + 1,
    }


def node_text_vector_search(state: AgentState) -> Dict:
    candidates = text_vectorstore.similarity_search_with_relevance_scores(
        state["query"], k=50, filter={"subject_id": state["subject_id"]}
    )
    results = []
    for doc, sim_score in candidates:
        date_str = doc.metadata.get("admittime", None)
        if date_str and state["allowed_years"]:
            time_weight = windowed_time_decay(date_str, state["allowed_years"])
        else:
            time_weight = 1.0
        results.append({"doc": doc, "final_score": sim_score * time_weight})

    results.sort(key=lambda x: x["final_score"], reverse=True)
    retrieved_docs_new = [r["doc"] for r in results[: state["num_retriev_text"]]]

    retrieved_docs_str = "\n\n".join(
        f"Document {i+1}:\n{textdoc2str(doc)}"
        for i, doc in enumerate(retrieved_docs_new)
    )
    retrieved_docs = state["retrieved_docs"]
    retrieved_docs.append(retrieved_docs_new)

    return {"retrieved_docs": retrieved_docs, "retrieved_docs_str": retrieved_docs_str}


def node_image_vector_search(state: AgentState) -> Dict:
    retrieved_docs_new = image_vectorstore.similarity_search(
        state["query"],
        k=state["num_retriev_img"],
        filter={"subject_id": state["subject_id"]},
    )
    retrieved_docs_str = "\n\n".join(imagedoc2str(doc) for doc in retrieved_docs_new)

    retrieved_docs = state["retrieved_docs"]
    retrieved_docs.append(retrieved_docs_new)

    return {"retrieved_docs": retrieved_docs, "retrieved_docs_str": retrieved_docs_str}


def node_update_summary(state: AgentState) -> Dict:
    current_summary = state["summary"]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT_UPDATE_TEMPLATE.format(
                        retrieved_docs_str=state["retrieved_docs_str"],
                        summary=current_summary,
                    ),
                }
            ],
        },
    ]
    response = PIPE(messages)
    patch = parse_response_json(response)
    if not isinstance(patch, dict):
        patch = {}
    return {"summary": apply_patch(current_summary, patch)}


def route(state: AgentState) -> str:
    return state["action"]


# --- Stage 2 nodes (non-streaming chat) ----------------------------------

def node_make_query(state: AgentState) -> Dict:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT_RAG_RETRIEVAL.format(
                        summary=state["summary"],
                        full_transcript=state["full_transcript"],
                        chat_history=state["chat_history"],
                        question=state["question"],
                    ),
                }
            ],
        },
    ]

    response = PIPE(messages, do_sample=False)
    plan = validate_plan(parse_response_json(response))
    if plan["action"] not in ("search_text", "search_imaging"):
        plan = {"action": "search_text", "query": state.get("question", ""), "allowed_years": 3}

    return {
        "action": plan["action"],
        "query": plan.get("query"),
        "allowed_years": plan.get("allowed_years"),
    }


def _build_answer_messages(state: AgentState) -> List[Dict[str, Any]]:
    """Construct the message list passed to the LLM for the chat-answer step.
    Shared between the non-streaming node and the streaming helper."""
    user_content = [
        {
            "type": "text",
            "text": (
                PROMPT_RAG.format(
                    question=state["question"],
                    full_transcript=state["full_transcript"],
                    retrieved_docs_str=state["retrieved_docs_str"],
                )
                if len(state["chat_history"]) > 0
                else START_CHAT_PROMPT.format(summary=state["summary"])
                + PROMPT_RAG.format(
                    question=state["question"],
                    full_transcript=state["full_transcript"],
                    retrieved_docs_str=state["retrieved_docs_str"],
                )
            ),
        }
    ]
    latest_docs = state["retrieved_docs"][-1] if state.get("retrieved_docs") else []
    user_content = add_images_to_user_content(user_content, latest_docs)

    if state.get("uploaded_image_path"):
        user_content = [{"type": "image", "url": state["uploaded_image_path"]}] + user_content

    user_prompt = {"role": "user", "content": user_content}
    return list(state.get("chat_history", [])) + [user_prompt]


def node_answer_question(state: AgentState) -> Dict:
    messages = _build_answer_messages(state)
    response = PIPE(messages, do_sample=False)
    answer_text = extract_assistant_text(response)
    messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
    )
    return {"chat_history": messages, "uploaded_image_path": None}


# --- Streaming helpers --------------------------------------------------

def chat_retrieve(state: AgentState) -> AgentState:
    """Run only the make_query + retrieval part of a chat turn so the app can
    stream the answer separately."""
    plan_update = node_make_query(state)
    state.update(plan_update)
    if state["action"] == "search_imaging":
        state.update(node_image_vector_search(state))
    else:
        state.update(node_text_vector_search(state))
    return state


def _strip_thinking_channel(source: Iterator[str]) -> Iterator[str]:
    """Yield deltas from an upstream stream, hiding Gemma 4's optional
    ``<channel|>thought<channel|>`` prefix from the user-visible output.

    Detection: peek at the first non-whitespace character of the stream.
    If it isn't ``<`` there's no thinking-channel header coming, so we
    flush and switch to immediate passthrough — this is the common case
    when vLLM serves Gemma 4 with thinking disabled, and the previous
    blanket-buffer logic made answers appear in one shot at end-of-stream
    instead of streaming."""
    pending = ""
    mode = "deciding"  # "deciding" | "in_thought" | "passthrough"
    for chunk in source:
        if mode == "passthrough":
            yield chunk
            continue

        pending += chunk

        if mode == "deciding":
            stripped = pending.lstrip()
            if not stripped:
                continue  # whitespace-only so far, keep peeking
            if stripped.startswith("<"):
                mode = "in_thought"
            else:
                mode = "passthrough"
                yield pending
                pending = ""
                continue

        # mode == "in_thought"
        if GEMMA4_THINK_SEP in pending:
            mode = "passthrough"
            _, after = pending.split(GEMMA4_THINK_SEP, 1)
            pending = ""
            if after:
                yield after
        elif len(pending) > 4000:
            # Safety net for a malformed long thinking block.
            mode = "passthrough"
            yield pending
            pending = ""
    if pending:
        yield pending


def stream_chat_answer(
    state: AgentState, max_new_tokens: int = 2000
) -> Iterator[str]:
    """Yield text deltas for the assistant's reply, hiding Gemma 4's
    optional thinking channel from the user-visible stream. Falls back to
    a single blocking generation if vLLM streaming isn't available."""
    messages = _build_answer_messages(state)
    try:
        yield from _strip_thinking_channel(
            PIPE.stream(messages, max_new_tokens=max_new_tokens)
        )
    except Exception:
        response = PIPE(messages, do_sample=False, max_new_tokens=max_new_tokens)
        yield extract_assistant_text(response)


def commit_chat_answer(state: AgentState, answer_text: str) -> Dict:
    """Append a streamed answer to the chat history. Mirrors what
    `node_answer_question` does at the end of a non-streaming turn."""
    messages = _build_answer_messages(state)
    messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
    )
    return {"chat_history": messages, "uploaded_image_path": None}


# --- Alerts and SOAP note (out-of-graph helpers) ------------------------

def compute_alerts(state: AgentState) -> List[Dict[str, str]]:
    """Run a single LLM call to surface clinically significant intersections
    between the patient summary and the live conversation."""
    transcript = (state.get("full_transcript") or "").strip()
    if not transcript:
        return []
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT_ALERTS.format(
                        summary=state["summary"],
                        full_transcript=transcript,
                        conversation_summary=state.get("conversation_summary", {}),
                    ),
                }
            ],
        },
    ]
    response = PIPE(messages, do_sample=False, max_new_tokens=1200)
    parsed = parse_response_json(response)
    alerts = parsed.get("alerts") if isinstance(parsed, dict) else None
    if not isinstance(alerts, list):
        return []
    cleaned = []
    for a in alerts:
        if not isinstance(a, dict):
            continue
        title = (a.get("title") or "").strip()
        rationale = (a.get("rationale") or "").strip()
        if not (title or rationale):
            continue
        cleaned.append(
            {
                "title": title or "(alert)",
                "rationale": rationale,
                "suggested_action": (a.get("suggested_action") or "").strip(),
            }
        )
    return cleaned


def generate_soap_note(state: AgentState) -> str:
    """Draft a SOAP note from the patient summary + live transcript +
    conversation summary. Returns markdown."""
    transcript = (state.get("full_transcript") or "").strip()
    if not transcript:
        return "_No conversation captured yet — start the live transcription first._"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT_SOAP_NOTE.format(
                        summary=state["summary"],
                        full_transcript=transcript,
                        conversation_summary=state.get("conversation_summary", {}),
                    ),
                }
            ],
        },
    ]
    response = PIPE(messages, do_sample=False, max_new_tokens=2500)
    return extract_assistant_text(response)


# --- Stage 1 graph -------------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("reason_and_plan", node_reason_and_plan)
graph.add_node("text_vector_search", node_text_vector_search)
graph.add_node("image_vector_search", node_image_vector_search)
graph.add_node("update_summary", node_update_summary)

graph.add_edge(START, "reason_and_plan")
graph.add_conditional_edges(
    "reason_and_plan",
    route,
    {
        "search_text": "text_vector_search",
        "search_imaging": "image_vector_search",
        "finish": END,
    },
)
graph.add_edge("text_vector_search", "update_summary")
graph.add_edge("image_vector_search", "update_summary")
graph.add_edge("update_summary", "reason_and_plan")

SUMMARY_AGENT = graph.compile()


# --- Stage 2 graph (non-streaming chat) ---------------------------------

graph = StateGraph(AgentState)

graph.add_edge(START, "make_query")
graph.add_node("make_query", node_make_query)
graph.add_node("text_vector_search", node_text_vector_search)
graph.add_node("image_vector_search", node_image_vector_search)
graph.add_node("answer_question", node_answer_question)

graph.add_conditional_edges(
    "make_query",
    route,
    {
        "search_text": "text_vector_search",
        "search_imaging": "image_vector_search",
    },
)
graph.add_edge("text_vector_search", "answer_question")
graph.add_edge("image_vector_search", "answer_question")
graph.add_edge("answer_question", END)

CHAT_TURN_AGENT = graph.compile()


def make_initial_state(subject_id: int) -> Dict[str, Any]:
    """Build a fresh agent state for a given patient. Templates are
    deep-copied so concurrent sessions don't share mutable structures."""
    return {
        "subject_id": int(subject_id),
        "action": "",
        "query": "",
        "allowed_years": None,
        "retrieved_docs": [],
        "retrieved_docs_str": "",
        "num_retriev_text": 3,
        "num_retriev_img": 2,
        "summary": deepcopy(SUMMARY_TEMPLATE),
        "action_history": [],
        "step": 0,
        "question": "",
        "chat_history": [],
        "uploaded_image_path": None,
        "audio_chunk": None,
        "transcript_chunk": "",
        "full_transcript": "",
        "conversation_summary": deepcopy(CONVERSATION_SUMMARY_TEMPLATE),
        "chunk_count": 0,
        "last_summary_at_len": 0,
    }


INITIAL_STATE = make_initial_state(13221453)
