# File descriptions

| File | Purpose |
|------|---------|
| `app.py` | Streamlit application: patient selector, summary generation, live transcription, chat with optional image upload. |
| `agent_demo.py` | LangGraph agents used by the app (`SUMMARY_AGENT` and `CHAT_TURN_AGENT`), the `make_initial_state(subject_id)` factory, and out-of-graph helpers `chat_retrieve` / `stream_chat_answer` / `compute_alerts` / `generate_soap_note`. |
| `audio_agent.py` | LangGraph transcription + conversation-summary agent (10-second chunks, 10 % overlap; structured summary re-runs every Nth chunk to keep latency reasonable). |
| `embeddings.py` | `MedSigLIPEmbeddings` (imaging) and `ClinicalBERTEmbeddings` (text) used by the FAISS vector stores. |
| `make_vdbs.py` | Builds text and imaging vector stores from every `data/patient_*_history.json`; also writes `data/vdbs/patients.json` (patient roster). |
| `models.py` | Loads the Gemma 4 image-text-to-text pipeline and the MedASR speech pipeline. Models and devices are configurable via `CLINICIAN_LLM`, `CLINICIAN_ASR`, `CLINICIAN_LLM_DEVICE`, `CLINICIAN_ASR_DEVICE`. |
| `prompts.py` | Prompt strings used in the agents. |
| `templates.py` | Patient-summary and conversation-summary JSON templates. |
| `utils.py` | Helpers: Gemma-4 response parsing (`parse_response_json` / `extract_assistant_text`), time-aware retrieval scoring, JSON-patch summary updates, image-path resolution. |
