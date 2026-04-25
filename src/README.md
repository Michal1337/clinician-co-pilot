# File Descriptions:
1. `agent_demo.py` - Agent Used in Demo
2. `agent.py`- Original Agents, only difference between `agent_demo.py` is one extra node in `CHAT_TURN_AGENT`
3. `app.py` - Actual Strealit Application
4. `audio_agent.py` - LangGraph Transcription and Summarization Agent
5. `embeddings.py` - Embedding Classes used in VectorStores 
6. `make_vdbs.py` - Create VectorStores from extracted data from MIMIC Datasets 
7. `models.py` - initalize LLMs and ASR
8. `prompts.py` - Prompts used in agents
9. `templates.py` - Patient Summary and Conversation Summary Template
10. `utils.py` - Helper function, time-awere retrieval, Apply patch to summary template
