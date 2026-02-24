import copy
import time

import librosa
import streamlit as st

from agent_demo import CHAT_TURN_AGENT, INITIAL_STATE, SUMMARY_AGENT
from audio_agent import AUDIO_AGENT, CHUNK_SAMPLES, STEP_SAMPLES


def render_stage1(state, query_action_box, summary_box):
    """Render current action/query and a patient summary, skipping empty sections.
    Evidence is rendered inline as subscript next to each statement.
    Handles both string and dict items safely.
    """

    # Show current action/query (UNCHANGED)
    if state.get("action") != "finish":
        query_action_box.markdown(
            f"### 🔎 Current State\n**Action:** `{state.get('action')}`  \n"
            f"**Query:** `{state.get('query')}`  \n"
            f"**Allowed Years:** `{state.get('allowed_years')}`"
        )
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
        for ev in get_list(item, "evidence"):
            if nonempty_str(ev.get("source_id", "")) or nonempty_str(
                ev.get("date", "")
            ):
                return True
        return False

    # 🔹 NEW: Inline evidence formatter
    def format_evidence_inline(evs):
        formatted = []
        for ev in evs:
            src = (ev.get("source_id") or "").strip()
            date = (ev.get("date") or "").strip()
            if not (src or date):
                continue
            if src and date:
                formatted.append(f"`{src}` · {date}")
            elif src:
                formatted.append(f"`{src}`")
            else:
                formatted.append(f"{date}")
        if not formatted:
            return ""
        joined = " | ".join(formatted)
        return f" <sub style='color:gray'>[{joined}]</sub>"

    md_lines = ["### 🧾 Patient Summary\n"]
    added_content = False

    def render_section(title, items, keys, formatter=None):
        nonlocal added_content
        filtered = [i for i in items if item_has_any_text(i, keys)]
        if not filtered:
            return
        added_content = True
        md_lines.append(f"#### {title}")
        for i in filtered:
            if isinstance(i, str):
                md_lines.append(f"- {i}")
            else:
                md_lines.append(formatter(i))
        md_lines.append("")

    # Active Problems
    def format_problem(p):
        prob = (p.get("problem") or "").strip() or "(unspecified problem)"
        status = (p.get("status") or "").strip()
        line = f"- **{prob}**" + (f" — {status}" if status else "")
        return line + format_evidence_inline(get_list(p, "evidence"))

    render_section(
        "⚠️ Active Problems",
        get_list(summary, "active_problems"),
        ["problem", "status"],
        format_problem,
    )

    # Recent Events
    def format_event(e):
        event_text = (e.get("event") or "").strip()
        date = (e.get("date") or "").strip()
        line = f"- {event_text}" + (f" — {date}" if date else "")
        return line + format_evidence_inline(get_list(e, "evidence"))

    render_section(
        "📝 Recent Events",
        get_list(summary, "recent_events"),
        ["event"],
        format_event,
    )

    # Medications
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

    render_section(
        "💊 Medications",
        get_list(summary, "medications"),
        ["name", "dose", "route"],
        format_med,
    )

    # Key Results
    def format_result(r):
        test = (r.get("test") or "").strip() or "(test)"
        result = (r.get("result") or "").strip()
        date = (r.get("date") or "").strip()
        line = f"- **{test}**"
        if result:
            line += f": {result}"
        if date:
            line += f" — {date}"
        return line + format_evidence_inline(get_list(r, "evidence"))

    render_section(
        "📊 Key Results",
        get_list(summary, "key_results"),
        ["test", "result"],
        format_result,
    )

    # Procedures
    def format_proc(p):
        proc = (p.get("procedure") or "").strip()
        date = (p.get("date") or "").strip()
        line = f"- {proc}" + (f" — {date}" if date else "")
        return line + format_evidence_inline(get_list(p, "evidence"))

    render_section(
        "🏥 Procedures",
        get_list(summary, "procedures"),
        ["procedure"],
        format_proc,
    )

    # Allergies
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

    render_section(
        "⚠️ Allergies",
        get_list(summary, "allergies"),
        ["substance", "reaction"],
        format_allergy,
    )

    # Pending Items
    def format_pending(p):
        item = (p.get("item") or "").strip()
        line = f"- {item}" if item else "- (unspecified)"
        return line + format_evidence_inline(get_list(p, "evidence"))

    render_section(
        "⏳ Pending Items",
        get_list(summary, "pending_items"),
        ["item"],
        format_pending,
    )

    if not added_content:
        summary_box.markdown(
            "### 🧾 Patient Summary\nNo populated sections in the summary yet."
        )
        return

    # allow HTML for subscript styling
    summary_box.markdown("\n".join(md_lines), unsafe_allow_html=True)


# ----------------- Audio streaming function -----------------
def run_audio_agent(audio_path, placeholder):
    """
    Stream audio chunks and update live transcript and conversation summary.
    Uses render_live_transcription() for consistent display.
    """
    time.sleep(10)
    # Load audio waveform
    waveform, sr = librosa.load(audio_path, sr=16000)
    total_samples = len(waveform)

    # Stream audio in steps
    for start in range(0, total_samples, STEP_SAMPLES):
        chunk = waveform[start : start + CHUNK_SAMPLES]
        st.session_state.state["audio_chunk"] = chunk

        # Stream through AUDIO_AGENT
        for event in AUDIO_AGENT.stream(st.session_state.state):
            for _, node_output in event.items():
                # Update session state
                st.session_state.state.update(node_output)

                # Render transcript + conversation summary in placeholder
                render_live_transcription(st.session_state.state, placeholder)


def render_live_transcription(state, placeholder):
    """
    Render live transcript and conversation summary together.
    Adds titles similar to render_stage1.
    """
    md_lines = []

    # Add transcript title and content
    md_lines.append("### 🎤 Live Transcript\n")
    transcript = state.get("full_transcript", "").strip()
    md_lines.append(transcript if transcript else "_No transcript yet._")

    # Add conversation summary title and content
    md_lines.append("\n### 🧾 Conversation Summary\n")
    conversation_summary = state.get("conversation_summary", {})
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

    # Render everything in the placeholder
    placeholder.markdown("\n".join(md_lines), unsafe_allow_html=True)


# ----------------- Page config & styling -----------------
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

st.title("🩺 Clinician Co-Pilot: AI Support Across the Patient Visit")

# ----------------- Session state -----------------
if "state" not in st.session_state:
    st.session_state.state = copy.deepcopy(INITIAL_STATE)
if "stage1_done" not in st.session_state:
    st.session_state.stage1_done = False
if "generating_summary" not in st.session_state:
    st.session_state.generating_summary = False
if "show_summary" not in st.session_state:
    st.session_state.show_summary = True  # default to show summary after stage1
if "live_transcription_started" not in st.session_state:
    st.session_state.live_transcription_started = False
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


# ----------------- Main UI -----------------
def main_ui():
    col_left, col_right = st.columns([6, 4])

    # ----------------- LEFT COLUMN -----------------
    with col_left:
        st.header("🩺 Patient Summary Generation / 🎙️ Live Transcription")

        # ----------------- View toggle -----------------
        view_mode = st.radio(
            "View",
            options=["Patient Summary Generation", "Live Transcription"],
            index=0,  # Default to Summary
            key="view_mode_radio",
            help="Choose to view the generated patient summary or the live transcription",
        )
        st.session_state.show_summary = view_mode == "Patient Summary Generation"

        # ----------------- Placeholders -----------------
        query_action_box = st.empty()
        summary_box = st.empty()
        live_placeholder = st.empty()

        # ----------------- Summary view -----------------
        if st.session_state.show_summary:
            # Show Generate button only if summary not done
            if not st.session_state.stage1_done:
                if st.button("Generate Summary", key="generate_summary_btn"):
                    st.session_state.generating_summary = True
                    with st.spinner("Generating Patient Summary..."):
                        for event in SUMMARY_AGENT.stream(st.session_state.state):
                            for node_name, node_output in event.items():
                                st.session_state.state.update(node_output)
                                # Render summary only after generation begins
                                render_stage1(
                                    st.session_state.state,
                                    query_action_box,
                                    summary_box,
                                )

                    st.session_state.stage1_done = True
                    st.session_state.generating_summary = False
                    st.session_state.show_summary = True

            # Render summary if already generated
            elif st.session_state.stage1_done:
                render_stage1(st.session_state.state, query_action_box, summary_box)

            # Clear live transcription placeholder when in Summary
            live_placeholder.empty()

        # ----------------- Live Transcription view -----------------
        else:
            # Clear summary placeholders when in Live Transcription
            query_action_box.empty()
            summary_box.empty()

            # Render live transcription and conversation summary
            if st.session_state.live_transcription_started:
                render_live_transcription(st.session_state.state, live_placeholder)

            # Start Recording button
            if st.button("Start Recording", key="start_record_btn"):
                st.session_state.live_transcription_started = True
                with st.spinner("Recording / streaming audio..."):
                    run_audio_agent("../data/conv.wav", live_placeholder)
                st.success("✅ Processing complete.")

    # ----------------- RIGHT COLUMN -----------------
    with col_right:
        st.header("💬 Clinical Chat")
        chat_placeholder = st.container()

        # Chat input form
        with st.form("chat_form", clear_on_submit=True):
            user_question = st.text_input("Ask about this patient")
            submitted = st.form_submit_button("Send", use_container_width=False)

        # Handle submission
        if submitted and user_question.strip():
            if not st.session_state.stage1_done:
                st.warning("Please generate the patient summary first (left column).")
            else:
                with st.spinner("Thinking..."):
                    st.session_state.state.update({"question": user_question})
                    st.session_state.state = CHAT_TURN_AGENT.invoke(
                        st.session_state.state
                    )

                    # Extract assistant answer
                    chat_history = st.session_state.state.get("chat_history", [])
                    assistant_answer = ""
                    if chat_history:
                        last_msg = chat_history[-1]
                        if last_msg.get("role") == "assistant":
                            assistant_answer = last_msg.get("content", [{"text": ""}])[
                                0
                            ].get("text", "")

                    st.session_state.qa_history.append(
                        {"question": user_question, "answer": assistant_answer}
                    )

        # Render chat history above input
        with chat_placeholder:
            if not st.session_state.qa_history:
                st.info(
                    "No chat history yet. Generate the patient summary and ask a question."
                )

            for qa in st.session_state.qa_history:
                # User bubble
                st.markdown(
                    f"""
                    <div style="
                        background-color:#1f77ff20;
                        padding:12px;
                        border-radius:12px;
                        margin-bottom:8px;
                        border:1px solid #1f77ff40;
                    ">
                        <b>You:</b><br>
                        {qa['question']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Assistant bubble
                st.markdown(
                    f"""
                    <div style="
                        background-color:#2e2e2e20;
                        padding:12px;
                        border-radius:12px;
                        margin-bottom:16px;
                        border:1px solid #99999930;
                    ">
                        <b>Agent:</b><br>
                        {qa['answer']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ----------------- Run UI -----------------
main_ui()
