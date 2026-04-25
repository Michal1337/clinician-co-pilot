import json
import warnings
from typing import Any, Dict, List, Literal, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from audio_agent import *
from embeddings import *
from models import PIPE
from prompts import *
from templates import *
from utils import *

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

    # Stage 1
    summary: Dict[str, Any]
    action_history: List[Dict[str, Any]]
    step: int

    # Stage 2
    question: str
    chat_history: List[str]

    # Stage 3
    audio_chunk: np.ndarray
    transcript_chunk: str
    full_transcript: str
    conversation_summary: str


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
    clean = extract_response_json(response[0]["generated_text"][-1]["content"])
    plan = json.loads(clean)
    print(plan)
    return {
        "action": plan["action"],
        "query": plan["query"],
        "allowed_years": plan.get("allowed_years", None),
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

        # Combine semantic + temporal
        final_score = sim_score * time_weight

        results.append({"doc": doc, "final_score": final_score})

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
    clean = extract_response_json(response)
    patch = json.loads(clean)
    updated_summary = apply_patch(current_summary, patch)
    return {"summary": updated_summary}


def route(state: AgentState) -> str:
    return state["action"]


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
    clean = extract_response_json(response[0]["generated_text"][-1]["content"])
    plan = json.loads(clean)

    return {
        "action": plan["action"],
        "query": plan["query"],
        "allowed_years": plan.get("allowed_years", None),
    }


def node_answer_question(state: AgentState) -> Dict:
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

    user_prompt = {
        "role": "user",
        "content": user_content,
    }

    messages = list(state.get("chat_history", []))
    messages.append(user_prompt)
    response = PIPE(messages, do_sample=False)

    assistant_reponse = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": response[0]["generated_text"][-1]["content"].split(
                    "<unused95>"
                )[1],
            }
        ],
    }

    messages.append(assistant_reponse)

    return {"chat_history": messages}


# Stage 1
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
# SUMMARY_AGENT.get_graph().draw_mermaid_png(output_file_path="../assets/summary_agent.png")

# Stage 2
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
# CHAT_TURN_AGENT.get_graph().draw_mermaid_png(output_file_path="../assets/chat_turn_agent.png")

INITIAL_STATE = {
    "subject_id": 13221453,
    "action": "",
    "query": "",
    "allowed_years": None,
    "retrieved_docs": [],  # list of retrieval-batches
    "retrieved_docs_str": "",
    "num_retriev_text": 1,
    "num_retriev_img": 1,
    "summary": SUMMARY_TEMPLATE,
    "action_history": [],
    "step": 0,
    "question": "",
    "chat_history": [],  # list of messages
    "audio_chunk": None,
    "transcript_chunk": "",
    "full_transcript": "",
    "conversation_summary": CONVERSATION_SUMMARY_TEMPLATE,
}
