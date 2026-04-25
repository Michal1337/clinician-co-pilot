# Clinician Co-Pilot: AI Support Across the Patient Visit

A local AI agent that summarizes patient records, imaging, and live visits to save physicians time and reduce documentation burden.

## Installation
1. Clone the repository:  `https://github.com/Michal1337/MedGemmaImpact.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Navigate to the source directory: `cd ./src`
4. Build the vector databases: `python make_vdbs.py`
5. Run the Streamlit app: `streamlit run app.py --client.toolbarMode minimal`

## Problem statement
Physicians today face a significant administrative burden that limits time spent with patients. For every hour of direct clinical face time, **nearly two additional hours are spent on EHR and desk work** during the clinic day. Outside office hours, physicians spend another 1–2 hours on personal time completing computer and clerical tasks (Sinsky et al., 2016). This imbalance reduces time available for patient interaction, increases cognitive load, and contributes to clinician burnout.

Patient information has become increasingly complex and multimodal. A single patient record may include longitudinal clinical notes, lab results, imaging studies such as X-rays, medication histories, specialist consultations, and discharge summaries. These data are often fragmented across multiple documents and formats, forcing physicians to manually integrate them before, during, and after visits. Clinicians must recall historical findings, interpret imaging, assess medications, and formulate diagnoses and treatment plans, all under time pressure, before completing structured documentation after the encounter.

Current EHRs and digital tools function primarily as passive storage or simple transcription aids. They do not synthesize multimodal data into structured summaries, allow natural language querying across both text and imaging, provide domain-specific medical reasoning, or integrate live conversation context. An AI system capable of performing these tasks could significantly reduce the time physicians spend on documentation and retrieval, potentially saving tens of minutes per patient, increasing face-to-face care time, improving diagnostic accuracy, and reducing cognitive overload. This makes the development of an intelligent clinical co-pilot highly impactful.

## Overall solution
Our solution is Clinician Co-Pilot, a multimodal AI agent designed to support physicians before, during, and after patient visits, reducing administrative burden and improving decision-making. By leveraging HAI-DEF models, the system can perform domain-specialized reasoning over both text and imaging data, providing accurate and context-aware clinical support. All models, are deployed locally, ensuring that sensitive patient data never leaves the hospital environment and fully addressing privacy and compliance concerns.

Before the visit, the agent automatically constructs a structured patient summary by retrieving information from two vector stores: one for textual documentation and one for imaging data. The imaging vector store uses MedSigLIP, allowing it to query X-rays using natural language. Each summary is generated in a consistent JSON format, with statements linked to source documents for transparency. This allows physicians to quickly understand a patient’s history, lab results, imaging findings, and medication profile without manually reviewing fragmented records.

During the visit, the agent focuses on live transcription and structured conversation summarization. When enabled, the system records the physician-patient dialogue and transcribes it in 10-second increments. These transcriptions are immediately incorporated into the conversation summary, capturing symptoms, clinical decisions, and discussion context in real time. Physicians can ask follow-up questions at any point, and the AI uses both historical records and the evolving live summary to provide accurate, context-aware responses. This creates a continuously updated, structured record of the encounter without disrupting the natural flow of conversation.

By combining local model deployment, multimodal retrieval, structured summarization, and live conversation capture, the system addresses inefficiencies in physician workflows. It can potentially save tens of minutes per patient, allowing more face-to-face time, reducing cognitive load, and supporting safer, faster clinical decisions.
