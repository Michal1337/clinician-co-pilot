# Review: Clinician Co-Pilot (Gemma-4-Good Hackathon Submission)

## 1. The problem you chose

The framing is strong and well grounded. The "2 hours of EHR work per 1 hour of patient face time" statistic from Sinsky et al. is the right hook — it positions the project against a measurable, well-documented pain point rather than a speculative one. The pitch correctly identifies three distinct workflow phases (pre-visit, intra-visit, post-visit) and maps each to a concrete agent capability, which gives the solution a coherent narrative rather than feeling like a tech demo searching for a use case.

The choice also plays well to the hackathon's specialized models: HAI-DEF / MedGemma / MedSigLIP / MedASR are all genuinely useful here, not bolted on. Local deployment as a privacy/compliance argument is also a real differentiator for a healthcare context — most cloud-LLM clinical demos hand-wave this away.

The main weakness in the framing is that the problem is huge and the demo necessarily covers a thin slice. That's appropriate for a hackathon, but the README slightly overpromises ("potentially saving tens of minutes per patient", "improving diagnostic accuracy") relative to what the running app actually does. A more honest framing would help: this is a *prototype of an architecture*, not a validated clinical tool.

## 2. The solution

### What works well

- **Three-agent decomposition** ([summary_agent.png](summary_agent.png), [chat_turn_agent.png](chat_turn_agent.png), [audio_agent.png](audio_agent.png)) is clean. Each agent has a clear job and a small state machine, which is much easier to reason about (and debug) than a single monolithic prompt-chain.
- **LangGraph** is a sensible choice for the agent loops — the `reason_and_plan → search → update_summary → reason_and_plan` cycle in [agent_demo.py:240-261](src/agent_demo.py#L240-L261) is the natural structure for the pre-visit task.
- **Two separate vector stores** ([make_vdbs.py](src/make_vdbs.py)) — ClinicalBERT for text and MedSigLIP for images — is the right call. Forcing one embedding to handle both modalities would have been worse.
- **Time-aware retrieval** ([utils.py:97-107](src/utils.py#L97-L107)) is a real touch of clinical realism. Most RAG demos ignore recency entirely; multiplying similarity by a windowed exponential decay and exposing `allowed_years` as a planner-controlled parameter is genuinely thoughtful.
- **Evidence linking** (`source_id` + `date` on every summary entry, rendered as inline subscripts in [app.py:59-75](src/app.py#L59-L75)) is exactly what a clinician would want. Provenance is a non-negotiable in this domain and it's wired through end-to-end.
- **JSON-patch summary updates** ([utils.py:110-140](src/utils.py#L110-L140)) instead of regenerating the whole summary each step. This is much more robust to LLM drift and makes per-step retrieval cheap.
- **Live transcription with rolling 10s chunks and 10% overlap** ([audio_agent.py:12-16](src/audio_agent.py#L12-L16)) is a reasonable streaming setup — and re-summarizing the running transcript into a structured `conversation_summary` rather than a free-text wall is the right shape for what a physician actually needs.
- **Strict no-repeat rule** in `PROMPT_REASON_AND_PLAN` ([prompts.py:36-40](src/prompts.py#L36-L40)) — small but important; without it the planner tends to loop on the same query.

### Where the solution is weaker

- **Image data is fetched but never shown to the user.** The agent retrieves X-ray documents, the chat model can technically see them via [utils.py:63-94](src/utils.py#L63-L94), and the embeddings store image paths — but `app.py` renders neither the image thumbnails nor any indication that an image was used. This is the single biggest gap given the multimodal premise of the project. (Discussed in detail under Improvements.)
- **Single hardcoded patient.** `subject_id: 13221453` is hardcoded in [agent_demo.py:289](src/agent_demo.py#L289) and [make_vdbs.py:10](src/make_vdbs.py#L10). The vector DB is only built for one patient. A working demo for the hackathon, but as written it cannot demonstrate the "scales to a hospital" claim.
- **Fragile JSON parsing.** [utils.py:8-15](src/utils.py#L8-L15) does a string `replace("null", "None")` and splits on the model token `<unused95>`. If the model omits the token (it does, sometimes), the bare except falls through silently. A schema-constrained decode (or `json_repair`) would be safer.
- **No test of summary correctness.** There is no eval harness, no gold-standard summary to compare against, no hallucination checks. This is fine for a hackathon but worth flagging in the writeup.
- **`agent.py` vs `agent_demo.py` split** is confusing — the [src/README.md](src/README.md) note that the only difference is "one extra node" is the kind of thing a reviewer trips over. Either delete `agent.py` or document why both exist.
- **`requirements.txt` is too thin.** It's missing `streamlit`, `torch`, `langchain-community`, `numpy`, `Pillow`, `python-dateutil`, etc. A reviewer who tries to run the project from a clean env will hit ImportErrors. Pin versions while you're at it.
- **`time.sleep(10)` at the top of `run_audio_agent`** ([app.py:229](src/app.py#L229)) is presumably a demo-pacing hack, but it's surprising and unexplained — leave a comment or remove it.
- **`max_new_tokens=32000` on the pipeline** ([models.py:9](src/models.py#L9)) is huge and will cost real latency on most steps. Consider reducing the default and overriding upward only in the few nodes that need it.
- **Hardcoded device IDs** (`cuda:2`, `cuda:3`) in [models.py](src/models.py) and [embeddings.py](src/embeddings.py) — reasonable on your dev box, but a barrier to anyone reproducing it.

## 3. Improvements (prioritized)

### High-impact

1. **Surface imaging in the UI (your own observation — strongly agree).** Three sub-improvements here, in increasing order of value:
   - **Display the X-ray.** When `node_image_vector_search` retrieves a study, render the JPEG inline in the patient summary section it influenced (or as a collapsible "Imaging evidence" panel under each summary item whose evidence cites an imaging `source_id`). Path is already in `doc.page_content` / `doc.metadata["main_image_path"]`.
   - **Let the chat agent show, not just see, images.** Today `add_images_to_user_content` in [utils.py:63-94](src/utils.py#L63-L94) puts images into the model's context, but the user has no idea. When the agent's answer cites an imaging document, render the corresponding image alongside the chat bubble in [app.py:446-461](src/app.py#L446-L461).
   - **Allow the user to upload an image.** A clinician viewing a fresh X-ray during a visit should be able to drop it into the chat and ask "compare to the patient's prior chest film" — your stack (MedSigLIP + Gemma-4 vision) already supports this; only the Streamlit `st.file_uploader` plumbing is missing.

2. **Multi-patient support.** Pull subject selection out of the constants:
   - Add a `st.selectbox` of `subject_id`s on app load, or read it from a query param.
   - Build the FAISS indices for *all* patients in `make_vdbs.py` (the metadata already filters by `subject_id`).
   - Reset `INITIAL_STATE` when the selected patient changes.

3. **Eval harness.** Even a small one:
   - Hand-write 5–10 reference summaries for known patients.
   - Score generated summaries against them on (a) coverage of true facts, (b) hallucination rate, (c) evidence-citation accuracy (does the cited `source_id` actually contain the claim?).
   - Citation accuracy is the most tractable and most impactful — you can check it programmatically by re-retrieving the `source_id` and string-matching key entities. This is also the metric a clinician audience cares about most.

4. **Robust structured-output parsing.** Replace the `<unused95>` split + regex strip with one of:
   - HuggingFace's `outlines` / `guidance` for grammar-constrained decoding,
   - `json_repair` as a fallback,
   - or at minimum, a retry loop ("your previous output failed JSON parse: <err>; return only JSON").

### Medium-impact

5. **Conversation summary should drive proactive prompts during the visit.** Right now the live transcript is passive — it updates `conversation_summary` and that's it. A clinician copilot earns its keep when it surfaces *during* the visit: "patient mentioned chest pain — last troponin was 6 months ago, want me to flag a workup?" This is one extra LangGraph node consuming `(conversation_summary, patient_summary)` and emitting an alert list.

6. **Post-visit note generation.** You have all the ingredients — pre-visit summary + live transcript + conversation summary + chat history. Add a fourth agent that drafts a SOAP note (or discharge note) from those four inputs. This closes the "before/during/after" loop the README promises but only the first two phases currently deliver.

7. **Streaming the LLM output.** Currently each agent node blocks until the full generation finishes; the Streamlit UI just spins. Streaming tokens (HF `TextIteratorStreamer`) would make the demo feel much more responsive — and matters more for a hackathon judge than for a real clinician.

8. **Better source rendering.** The evidence subscripts like `` `doc-uuid` · 2024-03-01 `` are functional but cryptic. Resolve `source_id → section name` ("Discharge summary, 2024-03-01") in [app.py:59-75](src/app.py#L59-L75); you already have the metadata.

9. **Guardrails on the planner's `allowed_years`.** The model sometimes omits it for non-historical queries despite the prompt rules. Cheap fix: validate the planner's JSON in `node_reason_and_plan` and re-prompt once if `allowed_years` is missing for a non-whitelisted query type.

### Low-impact / polish

10. **Remove the `time.sleep(10)`** in [app.py:229](src/app.py#L229) or comment it. It's a footgun for the next person who runs the demo.
11. **Delete `agent.py`** (or merge it with `agent_demo.py`) — the parallel files invite confusion.
12. **Pin and complete `requirements.txt`.** Add `streamlit`, `torch`, `numpy`, `Pillow`, `python-dateutil`, `langchain-community`, `pandas`, `datasets`, `tqdm`. Pin versions.
13. **Make device assignment configurable** via env vars (`MEDGEMMA_DEVICE`, `MEDASR_DEVICE`) instead of hardcoding `cuda:2` / `cuda:3`.
14. **Fix the chat history extraction** in [app.py:411-415](src/app.py#L411-L415) — chained `.get` with default `[{"text": ""}]` is fragile if the message structure ever changes.
15. **Add a "clear session" button** so judges can reset the demo without restarting Streamlit.
16. **Cap recursion of the summary agent** more visibly. `MAX_STEPS = 5` is enforced in [agent_demo.py:56-61](src/agent_demo.py#L56-L61) but not surfaced in the UI; show "step 3 of 5" so the user understands when generation will stop.

## 4. Bottom line

This is a strong submission. The architecture choices (multi-agent LangGraph, dual vector stores, time-decay retrieval, JSON-patch summaries, evidence-linked claims) are above the bar for a hackathon project — most of them are real engineering decisions, not just prompt-chain plumbing. The two areas that would most increase its impact are (1) actually showing the imaging the system has access to, and (2) backing the README's clinical claims with even a tiny eval. Everything else on the list is polish.
