PROMPT_REASON_AND_PLAN = (
    "You are a clinical retrieval planner building a pre-visit patient summary.\n\n"
    "CURRENT SUMMARY:\n{summary}\n\n"
    "PAST ACTIONS:\n{action_history}\n\n"
    "Pick ONE next action that fills the most important remaining gap.\n\n"
    "Actions:\n"
    "- search_text: query EHR notes / structured data (preferred).\n"
    "- search_imaging: query chest-X-ray reports (only when imaging is essential).\n"
    "- finish: stop when the summary is clinically sufficient or no high-value gap remains.\n\n"
    "Rules:\n"
    "- One action and ONE focused query per turn.\n"
    "- Never repeat or rephrase a past query — finish if no new gap exists.\n"
    "- Do NOT put time expressions in the query text. Control time via `allowed_years`.\n"
    "- Default windows: labs/meds 1-2y, chronic conditions 1-3y, imaging/procedures 2-5y.\n"
    "- Omit `allowed_years` only for inherently historical queries (initial diagnosis date,\n"
    "  past surgical history, genetic conditions).\n\n"
    "Return JSON only:\n"
    '{{"action": "search_text|search_imaging|finish", "query": "...", '
    '"allowed_years": <int or omit>, "thought": "..."}}'
)


PROMPT_UPDATE_TEMPLATE = (
    "NEWLY RETRIEVED DOCUMENTS:\n{retrieved_docs_str}\n\n"
    "CURRENT SUMMARY (JSON):\n{summary}\n\n"
    "Extract ONLY new clinical facts from the documents that are not already in the summary,\n"
    "and return them as a JSON patch grouped by section: active_problems, recent_events,\n"
    "medications, allergies, key_results, procedures, pending_items.\n\n"
    "Rules:\n"
    "- Do not rewrite, modify, or duplicate existing entries.\n"
    "- Place each fact in the correct section (active_problems = diagnoses; key_results = labs/vitals/imaging results).\n"
    '- Each new fact MUST include "evidence" with "source_id" and "date".\n'
    "- Use only explicit info from the documents — no inference.\n"
    "- If no new facts: return exactly {{}}.\n"
    "- Output strictly valid JSON, no commentary, no markdown."
)


START_CHAT_PROMPT = (
    "PATIENT CLINICAL SUMMARY:\n{summary}\n\n"
    "Answer the user using only the retrieved documents, the live conversation transcript,\n"
    "and the clinical summary. Prefer the transcript when it directly answers the question;\n"
    "otherwise rely on retrieved documents. Cite each claim with its Document ID and date."
)


PROMPT_RAG_RETRIEVAL = (
    "PATIENT CLINICAL SUMMARY:\n{summary}\n\n"
    "LIVE CONVERSATION TRANSCRIPT:\n{full_transcript}\n\n"
    "CHAT HISTORY:\n{chat_history}\n\n"
    "LATEST USER QUESTION:\n{question}\n\n"
    "Form ONE standalone medical search query optimised for vector retrieval.\n"
    "- Self-contained (no references to chat history).\n"
    "- Concise medical terminology.\n"
    "- No natural-language time expressions (use `allowed_years`).\n\n"
    "Actions: search_text (preferred) or search_imaging (only if imaging is essential).\n"
    "Default windows: labs/meds 1-2y, chronic 1-3y, imaging/procedures 2-5y.\n"
    "Omit `allowed_years` only for inherently historical queries.\n\n"
    "Return JSON only:\n"
    '{{"action": "search_text|search_imaging", "query": "...", "allowed_years": <int or omit>}}'
)


PROMPT_RAG = (
    "LIVE CONVERSATION TRANSCRIPT:\n{full_transcript}\n\n"
    "USER QUESTION: {question}\n\n"
    "RETRIEVED DOCUMENTS:\n{retrieved_docs_str}"
)


PROMPT_SUMMARIZE_TRANSCRIPTION = (
    "TRANSCRIPT:\n{full_text}\n\n"
    "CONVERSATION SUMMARY:\n{conversation_summary}\n\n"
    "Update the conversation summary based on the transcript.\n"
    "- Add new information; refine existing fields when the transcript clarifies them.\n"
    "- Do not remove confirmed facts unless explicitly corrected.\n"
    '- Add unanswered clinician questions to "open_questions".\n'
    "- Do not invent values. Output strictly valid JSON only."
)


PROMPT_ALERTS = (
    "PATIENT SUMMARY:\n{summary}\n\n"
    "LIVE TRANSCRIPT:\n{full_transcript}\n\n"
    "CONVERSATION SUMMARY:\n{conversation_summary}\n\n"
    "You are a clinical safety assistant. Surface only ALERTS the clinician should be aware of right now —\n"
    "clinically significant connections between the live conversation and the patient's history.\n\n"
    "Examples:\n"
    "- Patient mentions chest pain; last troponin was over 6 months ago.\n"
    "- Patient describes a new OTC medication that may interact with an existing prescription.\n"
    "- Reported symptom contradicts an active diagnosis in the summary.\n\n"
    "Rules:\n"
    "- Only include alerts SUPPORTED by both the conversation and the summary.\n"
    "- 0–5 alerts. If nothing is alert-worthy, return {{\"alerts\": []}}.\n"
    "- Each alert: short title + one-line rationale + (optional) suggested action.\n\n"
    "Return JSON only:\n"
    '{{"alerts": [{{"title": "...", "rationale": "...", "suggested_action": "..."}}]}}'
)


PROMPT_SOAP_NOTE = (
    "PATIENT SUMMARY:\n{summary}\n\n"
    "VISIT TRANSCRIPT:\n{full_transcript}\n\n"
    "CONVERSATION SUMMARY:\n{conversation_summary}\n\n"
    "Write a concise SOAP note in markdown for this visit.\n\n"
    "S — Subjective: chief complaint, HPI, relevant ROS in the patient's words.\n"
    "O — Objective: vitals, exam findings, lab/imaging results discussed in the visit.\n"
    "A — Assessment: differential or working diagnosis, with brief reasoning.\n"
    "P — Plan: medications, orders, follow-up, patient instructions.\n\n"
    "Rules:\n"
    "- Use only information present in the transcript or summary.\n"
    "- Do not invent vitals, dosages, or follow-up dates.\n"
    "- Mark uncertain items with \"(to confirm)\".\n"
    "- Output markdown only — no preamble, no JSON, no code fences."
)
