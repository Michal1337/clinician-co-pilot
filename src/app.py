import copy
import json
import os
import tempfile
import time

import soundfile as sf
import streamlit as st

from agent_demo import CHAT_TURN_AGENT, INITIAL_STATE, SUMMARY_AGENT
from audio_agent import AUDIO_AGENT, CHUNK_SAMPLES, STEP_SAMPLES
from utils import render_stage1


def run_audio_agent(audio_path, transcript_placeholder, summary_placeholder):
    waveform, sr = sf.read(audio_path)
    total_samples = len(waveform)

    for start in range(0, total_samples - CHUNK_SAMPLES + 1, STEP_SAMPLES):

        chunk = waveform[start:start + CHUNK_SAMPLES]
        st.session_state.state["audio_chunk"] = chunk

        for event in AUDIO_AGENT.stream(st.session_state.state):
            for _, node_output in event.items():
                st.session_state.state.update(node_output)

                # transcript_placeholder.text_area(
                #     label="",
                #     value=st.session_state.state.get("full_transcript", ""),
                #     height=250,
                #     key="live_transcript_box"
                # )
                transcript_placeholder.markdown(st.session_state.state.get("full_transcript", ""))
                summary_placeholder.json(
                    st.session_state.state.get("conversation_summary", {})
                )


st.set_page_config(layout="wide")
# Remove top padding and reduce font sizes
st.markdown("""
    <style>
        /* Remove top padding */
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
        }

        /* Remove extra margin above title */
        h1 {
            margin-top: 0rem !important;
            padding-top: 0rem !important;
            font-size: 28px !important;
        }

        h2 {
            font-size: 20px !important;
        }

        h3 {
            font-size: 16px !important;
        }

        /* Reduce button size */
        .stButton > button {
            padding: 0.3rem 0.8rem;
            font-size: 14px;
        }

        /* Reduce info box spacing */
        .stAlert {
            padding: 0.5rem 1rem;
        }
    </style>
""", unsafe_allow_html=True)
st.title("🩺 Multimodal Clinical Agent Demo")

# =====================================================
# SESSION STATE
# =====================================================

if "state" not in st.session_state:
    st.session_state.state = copy.deepcopy(INITIAL_STATE)
if "stage1_done" not in st.session_state:
    st.session_state.stage1_done = False

col1, col2, col3 = st.columns([3, 4, 3])  # 30% / 40% / 30%

# =====================================================
# COLUMN 1 — STAGE 1: PATIENT SUMMARY (LIVE)
# =====================================================

with col1:
    st.header("🩺 Patient Summary")

    query_action_box = st.empty()
    summary_box = st.empty()

    if st.button("Generate Summary", key="generate_summary"):
        st.session_state.generating_summary = True
        with st.spinner("Generating Patient Summary..."):
            for event in SUMMARY_AGENT.stream(st.session_state.state):
                for node_name, node_output in event.items():
                    st.session_state.state.update(node_output)
                    # print(st.session_state.state["summary"])
                    # print("@" * 50)                    
                    render_stage1(st.session_state.state, query_action_box, summary_box)

        st.session_state.stage1_done = True

# =====================================================
# COLUMN 2 — STAGE 2: CHAT
# =====================================================

with col2:
    st.header("💬 Clinical Chat")

    if not st.session_state.stage1_done:
        st.info("Generate the patient summary first.")
    else:

        # Chat display container
        chat_container = st.container(height=500)

        with chat_container:
            chat_history = st.session_state.state["chat_history"]

            for message in chat_history:
                role = message["role"]
                text = message["content"][0]["text"]

                if role == "user":
                    st.markdown(
                        f"""
                        <div style='background-color:#E8F0FE;padding:10px;border-radius:8px;margin-bottom:8px'>
                        <b>You:</b><br>{text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style='background-color:#F1F3F4;padding:10px;border-radius:8px;margin-bottom:8px'>
                        <b>Agent:</b><br>{text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Input area
        user_question = st.text_input("Ask about this patient")

        if st.button("Send"):
            if user_question.strip():
                with st.spinner("Thinking..."):
                    st.session_state.state.update({"question": user_question})
                    st.session_state.state = CHAT_TURN_AGENT.invoke(st.session_state.state)
                st.rerun()


# =====================================================
# COLUMN 3 — STAGE 3: AUDIO + SUMMARY
# =====================================================

with col3:
    st.header("💬 Live")
    if st.button("🎙️ Start Recording"):
        st.subheader("📝 Transcription")
        transcript_placeholder = st.empty()
        st.subheader("🧾 Conversation Summary")
        summary_placeholder = st.empty()

        run_audio_agent(
            "../test_audio.wav",
            transcript_placeholder,
            summary_placeholder
        )

        st.success("✅ Processing complete.")
