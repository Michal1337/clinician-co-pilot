import json
import os
import tempfile
import time
from pathlib import Path
from threading import Thread

import librosa
import streamlit as st

from agent_demo import (
    MAX_STEPS,
    SUMMARY_AGENT,
    chat_retrieve,
    commit_chat_answer,
    compute_alerts,
    generate_soap_note,
    image_vectorstore,
    make_initial_state,
    stream_chat_answer,
    text_vectorstore,
)
from audio_agent import AUDIO_AGENT, CHUNK_SAMPLES, SAMPLE_RATE, STEP_SAMPLES
from demographics import load_patient_demographics
from utils import (
    build_source_index,
    collect_image_paths,
    format_source_label,
)

ROSTER_PATH = Path("../data/vdbs/patients.json")


# ========================================================================
# Cached resources / data
# ========================================================================

@st.cache_data(show_spinner=False)
def load_subject_ids():
    if ROSTER_PATH.exists():
        try:
            with open(ROSTER_PATH, "r") as f:
                return [int(s) for s in json.load(f)]
        except Exception:
            pass
    ids = []
    for p in sorted(Path("../data").glob("patient_*_history.json")):
        try:
            with open(p, "r") as f:
                ids.append(int(json.load(f)["subject_id"]))
        except Exception:
            continue
    return ids


@st.cache_resource(show_spinner=False)
def get_source_index():
    return build_source_index(text_vectorstore, image_vectorstore)


@st.cache_data(show_spinner=False)
def _load_audio_cached(audio_path: str, mtime: float):
    waveform, _ = librosa.load(audio_path, sr=16000)
    return waveform


def find_conv_audio() -> str:
    for ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
        p = f"../data/conv{ext}"
        if os.path.exists(p):
            return p
    return "../data/conv.wav"


def load_audio(audio_path: str):
    return _load_audio_cached(audio_path, os.path.getmtime(audio_path))


@st.cache_data(show_spinner=False)
def get_demographics(subject_id: int):
    return load_patient_demographics(subject_id)


_ACTION_LABELS = {
    "search_text": "Searching clinical notes",
    "search_imaging": "Searching imaging studies",
    "finish": "Finalizing summary",
}


def _action_status_line(state) -> str:
    """One-line caption describing what the planner is doing right now."""
    action = state.get("action") or ""
    step = int(state.get("step") or 0)
    label = _ACTION_LABELS.get(action, action.replace("_", " ").title() if action else "")
    if not label:
        return ""
    line = f"Step {min(step, MAX_STEPS)}/{MAX_STEPS} · {label}"
    query = (state.get("query") or "").strip()
    if query and action != "finish":
        line += f" · _{query}_"
    return line


# ========================================================================
# Rendering helpers
# ========================================================================

def render_patient_header(demographics: dict, container) -> None:
    name = demographics.get("name") or "—"
    age = demographics.get("age")
    sex = demographics.get("sex")
    mrn = demographics.get("mrn") or ""
    age_sex = " · ".join(
        [f"{age} y/o" if age is not None else "", sex or ""]
    ).strip(" ·") or "Demographics pending"

    html = f"""
    <div style="
        background:#1f4e79;
        border-radius:6px;
        padding:10px 14px;
        margin:8px 0 14px 0;
        color:#ffffff;
        line-height:1.4;
    ">
      <div style="font-weight:600;font-size:18px;">{name}</div>
      <div style="font-size:15px;opacity:0.95;">{age_sex}</div>
      <div style="font-family:monospace;font-size:13px;opacity:0.85;margin-top:2px;">{mrn}</div>
    </div>
    """
    container.markdown(html, unsafe_allow_html=True)


def render_imaging_evidence(state, container, max_images=4):
    paths = collect_image_paths(state.get("retrieved_docs"), max_images=max_images)
    if not paths:
        container.empty()
        return
    with container.container():
        st.markdown("#### 🩻 Retrieved imaging")
        cols = st.columns(min(len(paths), max_images))
        for col, path in zip(cols, paths):
            try:
                col.image(path, width="stretch")
            except Exception as e:
                col.warning(f"Could not load {path}: {e}")


_ACTIVE_STATUS_KEYWORDS = (
    "active", "acute", "worsening", "new", "unstable", "decompensated", "uncontrolled",
)
_CHRONIC_STATUS_KEYWORDS = (
    "stable", "chronic", "resolved", "history of", "remote", "controlled", "well-controlled",
)
_REDUNDANT_STATUS = {"active", ""}
_UNKNOWN_DATE_TOKENS = {"unknown", "n/a", "none", "null", "?", "-", "--"}


def _clean_date(s) -> str:
    s = (s or "").strip()
    if s.lower() in _UNKNOWN_DATE_TOKENS:
        return ""
    return s


def _meaningful_status(s) -> str:
    s = (s or "").strip()
    if s.lower() in _REDUNDANT_STATUS:
        return ""
    return s


def _problem_priority(item) -> int:
    if not isinstance(item, dict):
        return 1
    s = (item.get("status") or "").lower()
    if any(k in s for k in _ACTIVE_STATUS_KEYWORDS):
        return 0
    if any(k in s for k in _CHRONIC_STATUS_KEYWORDS):
        return 2
    return 1


def render_stage1(
    state,
    query_action_box,
    summary_box,
    source_index=None,
    max_steps: int = 5,
    show_empty: bool = False,
    max_visible: int = 6,
):
    """Render planner state and the patient summary."""
    action = state.get("action")
    if action and action != "finish":
        line = _action_status_line(state)
        if line:
            query_action_box.info(line)
        else:
            query_action_box.empty()
    else:
        query_action_box.empty()

    summary = state.get("summary") or {}
    if not summary:
        summary_box.markdown("No summary yet.")
        return

    def nonempty_str(s):
        return isinstance(s, str) and s.strip() != ""

    def get_list(obj, key):
        val = obj.get(key)
        if val is None:
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            return [val]
        return []

    def item_has_any_text(item, keys):
        if isinstance(item, str):
            return nonempty_str(item)
        for k in keys:
            if nonempty_str(item.get(k, "")):
                return True
        return False

    def format_evidence_inline(evs):
        formatted = []
        for ev in evs:
            label = format_source_label(
                ev.get("source_id"),
                ev.get("date"),
                source_index=source_index,
            )
            if label:
                formatted.append(label)
        if not formatted:
            return ""
        return f" <sub style='color:gray'>[{' | '.join(formatted)}]</sub>"

    # Per-section formatters. Each returns one markdown bullet.
    def format_problem(p):
        prob = (p.get("problem") or "").strip() or "(unspecified problem)"
        status = _meaningful_status(p.get("status"))
        line = f"- **{prob}**" + (f" — {status}" if status else "")
        return line + format_evidence_inline(get_list(p, "evidence"))

    def format_event(e):
        event_text = (e.get("event") or "").strip()
        date = _clean_date(e.get("date"))
        line = f"- {event_text}" + (f" — {date}" if date else "")
        return line + format_evidence_inline(get_list(e, "evidence"))

    def format_med(m):
        name = (m.get("name") or "").strip()
        dose = (m.get("dose") or "").strip()
        route = (m.get("route") or "").strip()
        parts = [
            p
            for p in [
                name,
                (f"dose: {dose}" if dose else ""),
                (f"route: {route}" if route else ""),
            ]
            if p
        ]
        line = f"- {' — '.join(parts)}"
        return line + format_evidence_inline(get_list(m, "evidence"))

    def format_result(r):
        test = (r.get("test") or "").strip() or "(test)"
        result = (r.get("result") or "").strip()
        date = _clean_date(r.get("date"))
        line = f"- **{test}**"
        if result:
            line += f": {result}"
        if date:
            line += f" — {date}"
        return line + format_evidence_inline(get_list(r, "evidence"))

    def format_proc(p):
        proc = (p.get("procedure") or "").strip()
        date = _clean_date(p.get("date"))
        line = f"- {proc}" + (f" — {date}" if date else "")
        return line + format_evidence_inline(get_list(p, "evidence"))

    def format_allergy(a):
        sub = (a.get("substance") or "").strip()
        reaction = (a.get("reaction") or "").strip()
        if sub and reaction:
            line = f"- {sub} — {reaction}"
        elif sub:
            line = f"- {sub}"
        elif reaction:
            line = f"- (reaction) {reaction}"
        else:
            line = "- (unspecified)"
        return line + format_evidence_inline(get_list(a, "evidence"))

    def format_pending(p):
        item = (p.get("item") or "").strip()
        line = f"- {item}" if item else "- (unspecified)"
        return line + format_evidence_inline(get_list(p, "evidence"))

    # (title, summary_key, text_keys, formatter, empty_msg, sort_fn or None)
    sections = [
        ("⚠️ Active Problems", "active_problems", ["problem", "status"], format_problem,
         "No active problems documented in the available records.", _problem_priority),
        ("📝 Recent Events", "recent_events", ["event"], format_event,
         "No recent events surfaced.", None),
        ("💊 Medications", "medications", ["name", "dose", "route"], format_med,
         "No medications documented — confirm with patient.", None),
        ("📊 Key Results", "key_results", ["test", "result"], format_result,
         "No notable lab or test results retrieved.", None),
        ("🏥 Procedures", "procedures", ["procedure"], format_proc,
         "No prior procedures documented — does the patient recall any surgeries?", None),
        ("⚠️ Allergies", "allergies", ["substance", "reaction"], format_allergy,
         "NKDA / not documented — verify with patient.", None),
        ("⏳ Pending Items", "pending_items", ["item"], format_pending,
         "No outstanding follow-up items flagged.", None),
    ]

    def _fmt(item, formatter):
        return f"- {item}" if isinstance(item, str) else formatter(item)

    any_added = False
    with summary_box.container():
        st.markdown("### 🧾 Patient Summary")
        for title, key, text_keys, formatter, empty_msg, sort_fn in sections:
            items = get_list(summary, key)
            filtered = [i for i in items if item_has_any_text(i, text_keys)]

            if not filtered and not (show_empty and empty_msg):
                continue

            st.markdown(f"#### {title}")

            if not filtered:
                st.markdown(
                    f"<span style='color:#7b8794'>_{empty_msg}_</span>",
                    unsafe_allow_html=True,
                )
                continue

            any_added = True
            if sort_fn is not None:
                filtered = sorted(filtered, key=sort_fn)

            visible = filtered[:max_visible]
            hidden = filtered[max_visible:]

            st.markdown(
                "\n".join(_fmt(i, formatter) for i in visible),
                unsafe_allow_html=True,
            )

            if hidden:
                with st.expander(f"Show {len(hidden)} more"):
                    st.markdown(
                        "\n".join(_fmt(i, formatter) for i in hidden),
                        unsafe_allow_html=True,
                    )

        if not any_added and not show_empty:
            st.markdown("_No populated sections in the summary yet._")


def render_live_transcription(state, placeholder):
    md_lines = ["### 🎤 Live Transcript\n"]
    transcript = (state.get("full_transcript") or "").strip()
    md_lines.append(transcript if transcript else "_No transcript yet._")

    md_lines.append("\n### 🧾 Conversation Summary\n")
    conversation_summary = state.get("conversation_summary", {}) or {}
    added_content = False
    for key, value in conversation_summary.items():
        if isinstance(value, list) and value:
            md_lines.append(
                f"- **{key.replace('_', ' ').title()}:** {', '.join(map(str, value))}"
            )
            added_content = True
        elif isinstance(value, str) and value.strip():
            md_lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
            added_content = True
    if not added_content:
        md_lines.append("_No conversation summary yet._")

    placeholder.markdown("\n".join(md_lines))


def render_alerts(alerts, container):
    if not alerts:
        container.info("No clinical alerts to surface yet.")
        return
    with container.container():
        st.markdown("### 🚨 Clinical alerts")
        for a in alerts:
            title = a.get("title") or "(alert)"
            rationale = a.get("rationale") or ""
            action = a.get("suggested_action") or ""
            block = f"**{title}**"
            if rationale:
                block += f"  \n{rationale}"
            if action:
                block += f"  \n_Suggested:_ {action}"
            st.warning(block)


def run_audio_agent_threaded(waveform, total_samples: int, state: dict):
    # Pace each iteration to ~9 s of wall clock so transcript grows at
    # real conversation speed even when the GPU could finish faster.
    step_seconds = STEP_SAMPLES / SAMPLE_RATE
    for start in range(0, total_samples, STEP_SAMPLES):
        if state.get("audio_stop"):
            break
        tic = time.monotonic()
        chunk = waveform[start : start + CHUNK_SAMPLES]
        state["audio_chunk"] = chunk
        for event in AUDIO_AGENT.stream(state):
            for _, node_output in event.items():
                state.update(node_output)
        state["audio_progress"] = min(
            1.0, (start + CHUNK_SAMPLES) / max(total_samples, 1)
        )
        elapsed = time.monotonic() - tic
        remaining = step_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
    state["audio_progress"] = 1.0
    state["audio_done"] = True


@st.fragment(run_every="1s")
def render_live_panel_streaming():
    state = st.session_state.state
    placeholder = st.empty()
    render_live_transcription(state, placeholder)
    if state.get("audio_done"):
        st.rerun()


# ========================================================================
# Page setup + session state
# ========================================================================

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        .block-container { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
        h1 { margin-top: 0rem !important; font-size: 28px !important; }
        h2 { font-size: 24px !important; }
        h3 { font-size: 20px !important; }
        h4 { font-size: 16px !important; }
        h5 { font-size: 14px !important; }
        h6 { font-size: 12px !important; }
        .stButton > button { padding: 0.35rem 0.9rem; font-size: 14px; }
        .stAlert { padding: 0.5rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 Clinician Co-Pilot")

subject_ids = load_subject_ids()
if not subject_ids:
    st.error(
        "No patients available. Build the vector stores first: "
        "`python make_vdbs.py` from the `src/` directory."
    )
    st.stop()

with st.sidebar:
    st.header("Patient")
    selected = st.selectbox(
        "Subject ID",
        options=subject_ids,
        index=0,
        key="selected_subject_id",
        help="Choose the patient whose record this session should reason over.",
    )

    render_patient_header(get_demographics(int(selected)), st.sidebar)

    if st.button("Reset session", help="Clear summary, chat, and live transcription"):
        # Clean up uploaded image temp files before throwing away state.
        for qa in st.session_state.get("qa_history", []) or []:
            p = qa.get("user_image_path")
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        st.session_state.pop("state", None)
        st.session_state.pop("active_subject_id", None)
        st.rerun()


def _ensure_state_for(subject_id: int):
    if (
        "state" not in st.session_state
        or st.session_state.get("active_subject_id") != subject_id
    ):
        st.session_state.state = make_initial_state(subject_id)
        st.session_state.state["audio_done"] = False
        st.session_state.state["audio_progress"] = 0.0
        st.session_state.active_subject_id = subject_id
        st.session_state.stage1_done = False
        st.session_state.live_transcription_started = False
        st.session_state.qa_history = []
        st.session_state.alerts = []
        st.session_state.soap_note = None


_ensure_state_for(int(selected))

st.session_state.setdefault("stage1_done", False)
st.session_state.setdefault("live_transcription_started", False)
st.session_state.setdefault("qa_history", [])
st.session_state.setdefault("alerts", [])
st.session_state.setdefault("soap_note", None)

source_index = get_source_index()


def _save_uploaded_image(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(uploaded_file.getvalue())
        tmp.flush()
    finally:
        tmp.close()
    return tmp.name


# ========================================================================
# Main UI
# ========================================================================

def main_ui():
    col_left, col_right = st.columns([6, 4])

    with col_left:
        tab_summary, tab_visit = st.tabs(
            ["🩺 Patient Summary", "🎙️ Live Visit"]
        )

        with tab_summary:
            query_action_box = st.empty()
            summary_box = st.empty()
            imaging_box = st.empty()

            if not st.session_state.stage1_done:
                if st.button("Generate Summary", key="generate_summary_btn"):
                    with st.spinner("Generating patient summary…"):
                        for event in SUMMARY_AGENT.stream(st.session_state.state):
                            for _, node_output in event.items():
                                st.session_state.state.update(node_output)
                                render_stage1(
                                    st.session_state.state,
                                    query_action_box,
                                    summary_box,
                                    source_index=source_index,
                                    max_steps=MAX_STEPS,
                                )
                                render_imaging_evidence(
                                    st.session_state.state, imaging_box
                                )

                    st.session_state.stage1_done = True
                    st.rerun()
            else:
                render_stage1(
                    st.session_state.state,
                    query_action_box,
                    summary_box,
                    source_index=source_index,
                    max_steps=MAX_STEPS,
                    show_empty=True,
                )
                render_imaging_evidence(st.session_state.state, imaging_box)

        with tab_visit:
            if not st.session_state.live_transcription_started:
                if st.button("Start Recording", key="start_record_btn"):
                    st.session_state.live_transcription_started = True
                    st.session_state.state["audio_done"] = False
                    st.session_state.state["audio_progress"] = 0.0
                    waveform = load_audio(find_conv_audio())
                    thread = Thread(
                        target=run_audio_agent_threaded,
                        args=(waveform, len(waveform), st.session_state.state),
                        daemon=True,
                    )
                    thread.start()
                    st.session_state.audio_thread = thread
                    st.rerun()
            elif not st.session_state.state.get("audio_done"):
                render_live_panel_streaming()
            else:
                live_placeholder = st.empty()
                render_live_transcription(
                    st.session_state.state, live_placeholder
                )

                ctrl_cols = st.columns(2)
                alerts_box = st.empty()
                soap_box = st.empty()

                # Pre-populate before buttons so the other section stays
                # visible during a spinner.
                render_alerts(st.session_state.alerts, alerts_box)
                if st.session_state.soap_note:
                    with soap_box.container():
                        st.markdown("### 📝 SOAP note draft")
                        st.markdown(st.session_state.soap_note)

                with ctrl_cols[0]:
                    if st.button(
                        "Refresh alerts",
                        key="refresh_alerts_btn",
                        help="Run a single LLM pass to surface clinically significant "
                        "connections between the live conversation and the patient's "
                        "history.",
                        width="stretch",
                    ):
                        with st.spinner("Scanning for clinical alerts…"):
                            st.session_state.alerts = compute_alerts(
                                st.session_state.state
                            )
                        render_alerts(st.session_state.alerts, alerts_box)
                with ctrl_cols[1]:
                    if st.button(
                        "Generate SOAP note",
                        key="soap_btn",
                        help="Draft a SOAP note from the captured visit.",
                        width="stretch",
                    ):
                        with st.spinner("Drafting SOAP note…"):
                            st.session_state.soap_note = generate_soap_note(
                                st.session_state.state
                            )
                        with soap_box.container():
                            st.markdown("### 📝 SOAP note draft")
                            st.markdown(st.session_state.soap_note)

    with col_right:
        st.header("💬 Clinical Chat")
        chat_placeholder = st.container()

        with st.form("chat_form", clear_on_submit=True):
            user_question = st.text_input("Ask about this patient")
            uploaded_image = st.file_uploader(
                "Attach an image (optional)",
                type=["png", "jpg", "jpeg"],
                help="Drop a fresh X-ray or photo to ask about it directly.",
            )
            submitted = st.form_submit_button("Send")

        if submitted and user_question.strip() and st.session_state.stage1_done:
            handle_chat_submission(user_question, uploaded_image, chat_placeholder)
        else:
            if submitted and not st.session_state.stage1_done:
                st.warning("Please generate the patient summary first (left column).")
            render_chat_history(chat_placeholder)


def handle_chat_submission(user_question: str, uploaded_image, chat_placeholder):
    uploaded_path = _save_uploaded_image(uploaded_image) if uploaded_image else None
    st.session_state.state.update(
        {"question": user_question, "uploaded_image_path": uploaded_path}
    )

    with chat_placeholder:
        for qa in st.session_state.qa_history:
            _render_chat_pair(qa)
        with st.chat_message("user"):
            st.markdown(user_question)
            if uploaded_path and os.path.exists(uploaded_path):
                st.image(uploaded_path, caption="Your attachment", width=220)
        assistant_msg = st.chat_message("assistant")
        answer_placeholder = assistant_msg.empty()
        answer_placeholder.markdown("_Retrieving evidence…_")

    with st.spinner("Retrieving evidence…"):
        st.session_state.state = chat_retrieve(st.session_state.state)

    answer_text = ""
    with st.spinner("Answering…"):
        for delta in stream_chat_answer(st.session_state.state):
            answer_text += delta
            answer_placeholder.markdown(answer_text or "_…_")

    st.session_state.state.update(
        commit_chat_answer(st.session_state.state, answer_text)
    )

    used_paths = collect_image_paths(
        st.session_state.state.get("retrieved_docs", [])[-1:],
        max_images=3,
    )
    if uploaded_path:
        used_paths = [uploaded_path] + [p for p in used_paths if p != uploaded_path]

    qa = {
        "question": user_question,
        "answer": answer_text,
        "user_image_path": uploaded_path,
        "evidence_image_paths": used_paths,
    }
    st.session_state.qa_history.append(qa)

    evidence_paths = [
        p
        for p in used_paths
        if p and p != uploaded_path and os.path.exists(p)
    ]
    if evidence_paths:
        with assistant_msg:
            _render_chat_images(evidence_paths)


def _render_chat_images(paths):
    if not paths:
        return
    if len(paths) == 1:
        try:
            st.image(paths[0], width=220)
        except Exception as e:
            st.warning(f"Could not load {paths[0]}: {e}")
        return
    cols = st.columns(min(len(paths), 3))
    for col, p in zip(cols, paths):
        try:
            col.image(p, width="stretch")
        except Exception as e:
            col.warning(f"Could not load {p}: {e}")


def _render_chat_pair(qa: dict):
    with st.chat_message("user"):
        st.markdown(qa["question"])
        path = qa.get("user_image_path")
        if path and os.path.exists(path):
            st.image(path, caption="Your attachment", width=220)

    with st.chat_message("assistant"):
        st.markdown(qa.get("answer") or "_(no answer)_")
        evidence_paths = [
            p
            for p in qa.get("evidence_image_paths", []) or []
            if p and p != qa.get("user_image_path") and os.path.exists(p)
        ]
        _render_chat_images(evidence_paths)


def render_chat_history(chat_placeholder):
    with chat_placeholder:
        if not st.session_state.qa_history:
            st.info(
                "No chat history yet. Generate the patient summary and ask a question."
            )
            return
        for qa in st.session_state.qa_history:
            _render_chat_pair(qa)


main_ui()
