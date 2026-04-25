# Clinician Co-Pilot: AI Support Across the Patient Visit

A local AI agent that summarizes patient records, imaging, and live visits to save physicians time and reduce documentation burden. Built for the [Gemma-4-Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) on Kaggle.

## Installation
1. Clone the repository: `git clone https://github.com/Michal1337/MedGemmaImpact.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Pull patient histories from PhysioNet (one or more patients):
   ```bash
   cd ./data
   python get_data.py --subjects 13221453
   # or: python get_data.py --subjects-file my_subjects.txt
   ```
4. Build the vector databases over every extracted patient: `cd ../src && python make_vdbs.py`
5. Run the Streamlit app: `streamlit run app.py --client.toolbarMode minimal`

The app's sidebar exposes a **Subject ID** selector populated from whichever patients were extracted in step 3 — switching it resets the session for the new patient.

## Models
- **`google/gemma-4-26B-A4B-it`** — multimodal (text + imaging) reasoning, planning, summarization, and chat. Run locally on a GPU (`CLINICIAN_LLM_DEVICE`, default `cuda:2`).
- **`google/medsiglip-448`** — image-text embeddings for the chest-X-ray vector store, enabling natural-language search over imaging.
- **`google/medasr`** — speech-to-text for live visit transcription (`CLINICIAN_ASR_DEVICE`, default `cuda:3`).
- **`emilyalsentzer/Bio_ClinicalBERT`** — text embeddings for the clinical-notes vector store.

All models are deployed locally so sensitive patient data never leaves the hospital environment.

## Problem statement
Physicians today face a significant administrative burden that limits time spent with patients. For every hour of direct clinical face time, **nearly two additional hours are spent on EHR and desk work** during the clinic day. Outside office hours, physicians spend another 1–2 hours on personal time completing computer and clerical tasks (Sinsky et al., 2016). This imbalance reduces time available for patient interaction, increases cognitive load, and contributes to clinician burnout.

Patient information has become increasingly complex and multimodal. A single patient record may include longitudinal clinical notes, lab results, imaging studies such as X-rays, medication histories, specialist consultations, and discharge summaries. These data are often fragmented across multiple documents and formats, forcing physicians to manually integrate them before, during, and after visits. Clinicians must recall historical findings, interpret imaging, assess medications, and formulate diagnoses and treatment plans, all under time pressure, before completing structured documentation after the encounter.

Current EHRs and digital tools function primarily as passive storage or simple transcription aids. They do not synthesize multimodal data into structured summaries, allow natural language querying across both text and imaging, provide domain-specific medical reasoning, or integrate live conversation context. An AI system capable of performing these tasks could significantly reduce the time physicians spend on documentation and retrieval, potentially saving tens of minutes per patient, increasing face-to-face care time, improving diagnostic accuracy, and reducing cognitive overload. This makes the development of an intelligent clinical co-pilot highly impactful.

## Overall solution
Our solution is **Clinician Co-Pilot**, a multimodal AI agent designed to support physicians before, during, and after patient visits. By combining Gemma 4's general multimodal reasoning with domain-specialized embedding models (MedSigLIP for imaging, ClinicalBERT for notes) and MedASR for live transcription, the system performs accurate, context-aware clinical support entirely on local hardware.

**Before the visit**, the agent automatically constructs a structured patient summary by retrieving information from two vector stores: one for textual documentation and one for imaging data. The imaging vector store uses MedSigLIP, allowing it to query X-rays using natural language. Each summary is generated in a consistent JSON format, with statements linked to source documents for transparency. Retrieved X-rays are surfaced inline in the UI so the physician can see the actual imaging behind every claim.

**During the visit**, the agent focuses on live transcription and structured conversation summarization. When enabled, the system records the physician-patient dialogue and transcribes it in 10-second increments. These transcriptions are immediately incorporated into the conversation summary, capturing symptoms, clinical decisions, and discussion context in real time.

**Throughout the encounter**, physicians can ask follow-up questions in the chat panel, optionally attaching a fresh image (a new X-ray, a photo of a rash, etc.). The chat agent uses both historical records and the evolving live summary to provide accurate, context-aware responses; the answer streams token-by-token, and any imaging the agent referenced — retrieved from the vector store or supplied by the clinician — is rendered alongside the response. Evidence citations resolve raw document IDs to readable labels like "Discharge note · 2024-03-01".

**During and after the visit**, two on-demand actions help close the loop:

- *Refresh alerts* — surfaces clinically significant connections between the live conversation and the patient's history (e.g. patient mentions chest pain but last troponin is over six months old).
- *Generate SOAP note* — drafts a SOAP-formatted visit note from the patient summary, transcript, and conversation summary.

By combining local model deployment, multimodal retrieval, structured summarization, live conversation capture, clinician-supplied imaging, proactive in-visit alerts, and post-visit note drafting, the system addresses inefficiencies in physician workflows. It can potentially save tens of minutes per patient, allowing more face-to-face time, reducing cognitive load, and supporting safer, faster clinical decisions.
