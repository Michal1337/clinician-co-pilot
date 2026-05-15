"""Debug script for the pre-visit summary agent.

Streams ``SUMMARY_AGENT`` for a given patient and prints, at every node
event, exactly what came out: planner action/query, retrieved-doc count
and a peek at their content, the raw LLM response before parsing, the
parsed JSON patch, and the running summary's populated section counts.

Use this when ``Generate Summary`` in the app comes back blank — running
the same agent here surfaces *where* the chain breaks (planner returning
``finish`` immediately, retrieval returning 0 docs, the model's JSON
patch failing to parse, etc.).

    cd src
    python debug_summary.py                  # default subject 13221453
    python debug_summary.py --subject-id 10006580
    python debug_summary.py --subject-id 13221453 --raw     # dump raw model output too
"""

import argparse
import json
import sys
from pprint import pformat

from agent_demo import (
    MAX_STEPS,
    SUMMARY_AGENT,
    image_vectorstore,
    make_initial_state,
    text_vectorstore,
)
from models import LLM_SERVED_NAME, PIPE, VLLM_URL, _client as _vllm_client
from prompts import PROMPT_REASON_AND_PLAN
from utils import GEMMA4_THINK_SEP, parse_response_json


# --- pretty helpers -----------------------------------------------------

def hr(char="─", n=72):
    print(char * n)


def trunc(s: str, n: int = 400) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + f" …[+{len(s) - n} chars]"


def _item_populated(it) -> bool:
    if isinstance(it, str):
        return bool(it.strip())
    if isinstance(it, dict):
        # A template skeleton has all string fields blank. Treat the item as
        # populated if any non-evidence string field is non-empty.
        for k, v in it.items():
            if k == "evidence":
                continue
            if isinstance(v, str) and v.strip():
                return True
        return False
    return False


def summary_section_counts(summary: dict) -> dict:
    """How many populated items per section. Empty summary => all 0s."""
    if not isinstance(summary, dict):
        return {}
    out = {}
    for k, v in summary.items():
        if isinstance(v, list):
            out[k] = sum(1 for it in v if _item_populated(it))
        elif isinstance(v, str):
            out[k] = 1 if v.strip() else 0
        else:
            out[k] = "?"
    return out


# --- per-node printers --------------------------------------------------

def print_planner_event(state: dict, raw: bool):
    print("PLANNER")
    print(f"  step:          {state.get('step')}/{MAX_STEPS}")
    print(f"  action:        {state.get('action')!r}")
    print(f"  query:         {state.get('query')!r}")
    print(f"  allowed_years: {state.get('allowed_years')!r}")
    # history = state.get("action_history") or []
    # if history:
    #     last = history[-1]
    #     print(f"  last decision: {trunc(pformat(last), 300)}")


def print_retrieval_event(state: dict):
    docs = state.get("retrieved_docs") or []
    last = docs[-1] if docs else []
    print("RETRIEVAL")
    print(f"  total batches so far: {len(docs)}")
    print(f"  last batch size:      {len(last)}")
    if not last:
        print("  ⚠ NO DOCUMENTS — subject_id filter mismatch, "
              "empty vector store, or no matches for this query")
    for i, doc in enumerate(last[:3]):
        meta = doc.metadata or {}
        section = meta.get("section") or "(imaging)"
        admittime = meta.get("admittime") or meta.get("study_date") or "?"
        subj = meta.get("subject_id")
        preview = trunc(getattr(doc, "page_content", ""), 200)
        print(f"  doc[{i}] subj={subj} section={section} date={admittime}")
        print(f"          content: {preview}")
    if len(last) > 3:
        print(f"  …({len(last) - 3} more)")


def print_update_event(state: dict):
    """update_summary just sets `summary`. Show what's populated now."""
    counts = summary_section_counts(state.get("summary") or {})
    print("UPDATE SUMMARY")
    print(f"  section item counts: {counts}")
    nonzero = {k: v for k, v in counts.items() if isinstance(v, int) and v > 0}
    if not nonzero:
        print("  ⚠ summary still empty — patch likely failed to parse "
              "or didn't add any items")


def print_event(node: str, state: dict, raw: bool):
    # hr("·")
    # print(f"NODE: {node}")
    if node == "reason_and_plan":
        print_planner_event(state, raw=raw)
    # elif node in ("text_vector_search", "image_vector_search"):
    #     print_retrieval_event(state)
    # elif node == "update_summary":
    #     print_update_event(state)
    # else:
    #     print(trunc(pformat({k: state.get(k) for k in state if k != "audio_chunk"}), 800))


# --- vLLM connectivity check -------------------------------------------

def vllm_sanity() -> bool:
    """Verify the vLLM endpoint is up and serving the expected model name.
    Returns True on success; prints a diagnostic and returns False on
    failure so callers can short-circuit before doing real work."""
    hr("=")
    print("vLLM CONNECTIVITY CHECK")
    print(f"  endpoint:    {VLLM_URL}")
    print(f"  served name: {LLM_SERVED_NAME}")
    try:
        models = _vllm_client.models.list()
        ids = [m.id for m in models.data]
        print(f"  available:   {ids}")
        if LLM_SERVED_NAME not in ids:
            print(
                f"  ⚠ '{LLM_SERVED_NAME}' is not in the served-model list.\n"
                f"    Either start vLLM with --served-model-name {LLM_SERVED_NAME}\n"
                f"    or set CLINICIAN_LLM_SERVED to one of: {ids}"
            )
            return False
    except Exception as e:
        print(
            f"  ⚠ Could not reach vLLM at {VLLM_URL}: {e!r}\n"
            f"    Start the server first, e.g.:\n"
            f"      CUDA_VISIBLE_DEVICES=0 vllm serve {LLM_SERVED_NAME} \\\n"
            f"          --port 8000 --max-model-len 32768 --dtype bfloat16"
        )
        return False
    print("  ✓ ok")
    return True


# --- vector store sanity check -----------------------------------------

def vstore_sanity(subject_id: int):
    """Hit each vector store with a generic query and count subject-filtered
    matches. Helps distinguish 'no patient data indexed' from 'agent never
    asked for it'."""
    hr("=")
    print("VECTOR STORE SANITY CHECK")
    try:
        text_hits = text_vectorstore.similarity_search(
            "history", k=5, filter={"subject_id": subject_id}
        )
        print(f"  text_vdb hits for subject={subject_id}: {len(text_hits)}")
        for d in text_hits[:2]:
            print(f"    · section={d.metadata.get('section')} "
                  f"admittime={d.metadata.get('admittime')}")
    except Exception as e:
        print(f"  text_vdb query failed: {e!r}")

    try:
        img_hits = image_vectorstore.similarity_search(
            "chest", k=5, filter={"subject_id": subject_id}
        )
        print(f"  img_vdb  hits for subject={subject_id}: {len(img_hits)}")
        for d in img_hits[:2]:
            print(f"    · study_date={d.metadata.get('study_date')} "
                  f"path={d.metadata.get('main_image_path')}")
    except Exception as e:
        print(f"  img_vdb query failed: {e!r}")


# --- raw planner probe --------------------------------------------------

def raw_planner_probe(state: dict):
    """Run *one* planner step manually and dump the raw model output before
    JSON parsing — the most common silent failure is the model emitting
    text that doesn't survive ``parse_response_json``."""
    hr("=")
    print("RAW PLANNER PROBE (one shot, no graph)")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT_REASON_AND_PLAN.format(
                        summary=state["summary"],
                        action_history=state["action_history"],
                    ),
                }
            ],
        },
    ]
    response = PIPE(messages, do_sample=False, max_new_tokens=2000)
    # Try to pull the assistant text out of the HF pipeline response
    try:
        assistant_blob = response[0]["generated_text"]
        if isinstance(assistant_blob, list):
            text = next(
                (
                    c.get("content")
                    for c in reversed(assistant_blob)
                    if c.get("role") == "assistant"
                ),
                "",
            )
            if isinstance(text, list):
                text = " ".join(
                    p.get("text", "") for p in text if isinstance(p, dict)
                )
        else:
            text = assistant_blob
    except Exception as e:
        text = f"<could not extract: {e!r}>"
    print(f"  raw assistant text ({len(str(text))} chars):")
    print("  " + trunc(text, 1500).replace("\n", "\n  "))

    if GEMMA4_THINK_SEP not in str(text):
        print(f"  ⚠ No '{GEMMA4_THINK_SEP}' separator found in output — "
              "Gemma channel-split parsing will treat the whole thing as "
              "the thinking channel and skip it.")
    parsed = parse_response_json(response)
    print(f"  parsed JSON: {parsed!r}")
    if not isinstance(parsed, dict) or not parsed:
        print("  ⚠ parse_response_json returned empty/non-dict — "
              "this is why downstream nodes don't progress.")


# --- main ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-id", type=int, default=13221453)
    ap.add_argument("--raw", action="store_true",
                    help="Also do a one-shot raw-model probe of the planner.")
    ap.add_argument("--no-stream", action="store_true",
                    help="Skip the full agent stream (only the sanity check + raw probe).")
    args = ap.parse_args()

    if not vllm_sanity():
        return 1

    state = make_initial_state(args.subject_id)
    vstore_sanity(args.subject_id)

    if args.raw:
        raw_planner_probe(state)

    if args.no_stream:
        return

    hr("=")
    print(f"STREAMING SUMMARY_AGENT for subject_id={args.subject_id}")
    hr("=")

    n_events = 0
    for event in SUMMARY_AGENT.stream(state):
        for node, node_output in event.items():
            state.update(node_output)
            n_events += 1
            print_event(node, state, raw=args.raw)
    hr("=")
    print(f"DONE. {n_events} node events processed.")
    print("\nFinal summary section counts:")
    print(json.dumps(summary_section_counts(state.get("summary") or {}), indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
