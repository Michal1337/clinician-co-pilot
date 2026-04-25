# Ideas: making Clinician Co-Pilot land harder

The previous [REVIEW.md](REVIEW.md) was about correctness and completeness — the migration to Gemma 4 and the high/medium/low improvement rounds closed nearly everything on it. This document is the next layer: how do we make the submission *land* with a judge and feel like a real clinical product, not a clean tech demo? Four axes: **story**, **agent capability**, **UI**, and **clinical realism**.

Items are tagged with rough effort: **S** (an evening), **M** (a weekend), **L** (multi-day rebuild). Most should be cheap because the underlying agent stack already supports them.

---

## 1. Story and pitch

The current pitch is structurally fine ("before / during / after"), but it's abstract. A judge watches dozens of submissions; the ones that stick have a *protagonist*, a *clock*, and a *concrete win*. Right now the demo is "click Generate Summary on patient 13221453."

### 1.1 Frame the demo as a single 90-second visit (S)
Open the README and the demo video with a vignette:

> *8:52 AM. Dr. Patel has 14 patients today. Next up at 9:00: Mrs. Garcia, 67, readmission #3 for shortness of breath. Last time he saw her was four months ago; in between she's had two ED visits, a new diuretic, and a chest CT he hasn't read yet.*

Then every UI element on screen earns its place against that scene. The summary panel is "what Dr. Patel would have spent 12 minutes pulling from the chart." The alerts button is "what he would have missed." The SOAP note is "the 8 minutes after the visit that he gets back."

This is a README rewrite plus a 30-second voiceover, not code. But it's the single highest-leverage change.

### 1.2 Quantify the saved-minutes claim with a real (small) measurement (M)
The README says "tens of minutes per patient." A judge will ask *how many*. Pick three patient records, time how long it takes a non-clinician reviewer to write a structured summary by hand vs reading the generated one, and put a single number on the splash screen: **"Median 9 minutes saved per pre-visit chart review (n=3 patients)."** Caveat heavily — but a *small honest number* is more credible than a big hand-waved one.

### 1.3 Pick a clinical specialty and lean in (S)
Generic "physician copilot" is unfocused. The actual data (MIMIC-IV + chest X-rays) skews ICU/cardiopulmonary. Reframe as **"Co-Pilot for the cardiology readmission clinic"** or **"Co-Pilot for the ICU rounding team"**. The stack doesn't change; the prompts and the demo patient narrative do. Specificity reads as competence.

### 1.4 Add one believable failure mode and how the system handles it (S)
Demos that show only happy paths feel like vaporware. Show one case where:
- the agent says "I don't have enough evidence to summarize this section" (coverage gap), or
- the alerts panel correctly flags a contradiction in the patient summary the agent itself produced.

This is more persuasive to a clinical audience than five more polished features.

---

## 2. Agent capability — what to add, what to change

The current agent surface is: pre-visit summary, RAG chat, live transcript + summary, alerts, SOAP note. That covers the workflow but has gaps where a clinician's attention actually lives. Listed below in rough order of clinical leverage.

### 2.1 Differential diagnosis with reasoning trace (M)
Add a **"Generate differential"** button next to *Generate SOAP note*. Input: the patient summary + the live conversation summary. Output: a ranked list of `{diagnosis, supporting_evidence: [...], contradicting_evidence: [...], next_test_to_disambiguate}`. This is exactly what a clinician's brain does silently during a visit, and it's the kind of artifact the EHR can't produce. Gemma 4 is plenty strong for this — the trick is in the prompt and the JSON schema.

This is a much higher-impact use of the model than chat — a clinician will *use* a differential; they will rarely sit and chat with their EHR.

### 2.2 Order suggestions / next-step recommendations (M)
A natural sibling to the differential. Given the live conversation, suggest concrete next-step orders — "consider repeat troponin", "BNP", "CXR PA/lat", "echo if not done in 6 months". Constrained to a small whitelist of common orders so it can't hallucinate exotic ones. Render as checkboxes the clinician can tick into the SOAP note's *Plan* section.

### 2.3 Pre-visit "questions to ask" (S)
Before the visit, in addition to the summary, generate **5–10 specific questions** for this patient *based on what's missing or stale in their chart*: "Has the new diuretic helped your dyspnea?", "Any episodes of chest pain since the November ED visit?", "Are you still taking the carvedilol?". This is the most-requested feature from real clinicians piloting copilots — a printable cheat sheet — and it reuses the planner essentially unchanged.

### 2.4 Drug interaction / contraindication check (S)
Trivial extra LLM call given the medication list already extracted. Cross-check meds against the active problem list and flag classics: NSAID + CKD, BB + asthma, anticoag + recent GI bleed. Even 80% accuracy here is a clinical *talking point*. Render as a third row in the alerts panel.

### 2.5 Patient-facing summary in plain language (S)
Same patient summary, but rewritten at 6th-grade reading level for the after-visit handout: "You have heart failure, which means your heart isn't pumping as well as it should. Today we changed your water pill from 20mg to 40mg…". Clinicians spend real time on this; it's a one-prompt feature.

### 2.6 ICD-10 / CPT coding hints (M)
After the visit, suggest the diagnosis codes (ICD-10) and procedure codes (CPT/E&M level) based on the SOAP note. This is *the* unsexy financial reality of US medicine — every clinician does this every visit, hates it, and would pay for an AI that does it well. A short whitelist of the top 200 ICD-10 codes covers most of internal medicine.

### 2.7 Replace the freeform chat with a "Chart Q&A" verb-first action (M, controversial)
Honest assessment: the typed chat panel is the least clinically realistic part of the demo. Doctors don't type questions to their EHR. They'd rather have:

- a **voice-activated** "Hey copilot, when was the last echo?" — your MedASR pipeline is already there;
- or a **structured action menu** ("Show me the trend of [creatinine / BNP / BP] over [3mo / 1yr]") that maps to deterministic retrieval, not chat.

Keep chat as an "ask anything" escape hatch but make voice or structured actions the default. This is the single change that makes the demo *feel* clinical instead of generic-LLM-y.

### 2.8 Per-claim confidence / coverage indicators (S)
Already-extracted fields could carry a `confidence: low/med/high` marker derived from how many distinct documents support them. Render low-confidence claims in italic-gray and a question-mark icon the clinician can click to see the supporting docs. This is closer to how a senior physician reads a chart — *with skepticism* — than the current crisp summary suggests.

### 2.9 Compare-to-prior imaging panel (M)
When the agent surfaces a chest X-ray, automatically retrieve the *prior* X-ray for the same patient (next nearest-date imaging doc filtered by `subject_id`) and show them side-by-side. Even without an explicit "find changes" prompt, the side-by-side is what a radiologist's eye does first. With an extra prompt ("describe interval changes") it becomes a real differentiator.

### 2.10 Audit trail / "show me how you got here" (S)
Every summary claim and every alert should be reproducibly traceable. Add a small *Show reasoning* expander under each summary section that prints the planner's `action_history` for that section. Clinically this is the difference between "AI told me to do X" and "AI told me to do X *because* it found Y in note Z" — which matters for both adoption and medico-legal exposure.

---

## 3. UI / UX

The current 2-column Streamlit layout (summary | chat) is functional but reads as "RAG demo with a chat panel." Real EHRs are chart-first, timeline-shaped, and dense. Some shifts that are cheap in Streamlit and some that aren't.

### 3.1 Patient header strip (S)
Across the top of the page, always visible: **Name, Age, Sex, MRN, primary problem, allergy badges, code status**. Right now the patient is "Subject ID: 13221453" in a sidebar. Even using fake names (`Patient_13221453 → "Garcia, M."`) and a fixed avatar circle changes the whole tone of the demo. Clinicians orient to the patient banner before anything else.

### 3.2 Timeline view of the patient record (M)
A horizontal timeline component at the top of the summary panel — admissions as bars, ED visits as dots, imaging studies as little camera icons, medications as colored ribbons. Click a marker → that document scrolls into view as evidence. This is the single visual feature that most distinguishes a "real chart" from an "AI summary." Streamlit can do this with `plotly` or `altair` in <100 lines.

### 3.3 Editable summary fields (M)
Every field in the patient summary should be inline-editable. The clinician's mental loop is *read AI summary → correct one wrong fact → keep going*; if the only response to a wrong fact is "ignore it," the AI is just noise. Persist edits to the SOAP note input. (Streamlit's `st.data_editor` makes this nearly free for the list-of-dicts shape the summary already has.)

### 3.4 Replace the radio toggle between Summary / Live with a tabbed layout (S)
The current "View: [Patient Summary Generation] [Live Transcription]" radio reads as toggle between *two demos*. In practice both should be visible during a visit — summary on the left, transcript on the right, chat on the bottom. Streamlit `st.tabs` or just a 3-column layout. The current code already cleanly separates these — purely a layout change.

### 3.5 Streaming everywhere (S)
The chat already streams. The summary agent does not — Generate Summary spins for ~30 seconds and dumps the result. Stream the *summary updates* node-by-node with a "currently retrieving X" status line, so the user sees the agent thinking. The graph already emits per-node events; pipe them into a status placeholder (most of the plumbing is there in `render_stage1`).

### 3.6 "I don't have evidence for this" empty state (S)
When a section of the summary template comes back empty, the current UI just hides it. Replace with a faint placeholder: *"No documented procedures in the available records — does the patient remember any surgeries? [Add manually]"*. Empty-but-acknowledged is more clinically honest than silently absent.

### 3.7 Keyboard shortcuts and density (S)
Clinicians fly through EHRs on keyboard. Add the obvious ones: `g s` for summary, `g t` for transcript, `g c` for chat, `Cmd+Enter` to send chat, `r` to refresh alerts. Streamlit can hook these via a small JS component or `streamlit-shortcuts`. Tiny change, very large credibility increase for anyone who's used Epic.

### 3.8 Dark mode and clinical color palette (S)
Default Streamlit colors (orange accents, blue links) read as "consumer SaaS." Real clinical UIs are gray + one accent + red for critical. Add a theme file (`.streamlit/config.toml`) with neutral grays, navy primary, red for alerts only. Free 30% credibility.

### 3.9 Mobile / tablet form factor (M)
Real bedside use is on a tablet. Streamlit isn't great at this but a "compact mode" toggle that collapses to a single column with cards would make the rounding-on-iPad scenario plausible. Worth doing only if you're willing to rebuild in something more layout-flexible (Gradio Blocks, or a small React frontend over a FastAPI backend).

---

## 4. Clinical realism

The hardest gap to close, but the highest payoff for a healthcare-track judge. Three categories: **data shape**, **workflow integration**, and **medico-legal**.

### 4.1 FHIR-shaped output, not bespoke JSON (M)
The patient summary template is custom JSON. Every modern healthcare integration speaks **FHIR R4**. Re-shape the summary as a FHIR `Bundle` containing `Condition`, `MedicationStatement`, `Observation`, `AllergyIntolerance`, `Procedure`, `DiagnosticReport`. The fields are nearly identical to what the template already has — this is a renaming pass, not a redesign — but it's the difference between *toy data structure* and *plausibly integrable*.

Bonus: add a **"Send to EHR"** button that POSTs the FHIR bundle to a local mock endpoint (e.g. [HAPI FHIR](https://hapifhir.io/) Docker) and shows a confirmation. Costs an evening, looks like real integration.

### 4.2 Multi-encounter chart, not single blob (M)
Right now a patient's history is one flat document set. Real charts are organized by **encounter**: each admission/ED visit/clinic visit is its own folder with notes, labs, imaging, meds. The MIMIC-IV `hadm_id` is already in your metadata; group documents by it and render the timeline (3.2) by encounter, not by document.

### 4.3 Mock clinician identity + audit log (S)
Add a fake login at app start: *"Logged in as: Dr. R. Patel, Internal Medicine."* Every agent action writes to a visible audit log: *"08:53 — Dr. Patel viewed summary. 08:54 — Co-Pilot retrieved Discharge note 2024-03-01. 08:54 — Co-Pilot generated alert: stale troponin."* Clinicians and compliance officers immediately recognize this as *an actual healthcare app*; without it, it reads as a Jupyter notebook with a UI.

### 4.4 Disclaimers and medico-legal framing (S)
Every AI-generated section needs a small disclaimer footer: *"Generated by Co-Pilot 0.1 · Reviewed by clinician: ☐ · Not a substitute for clinical judgment"*. The SOAP note especially needs an **"Attest and sign"** checkbox before it can be exported. This is exactly the kind of detail that shows you've thought about deployment, not just inference.

### 4.5 Drift / staleness indicators on retrieved evidence (S)
A summary that cites a 4-year-old document as evidence for *current* problems should visibly flag that. The time-decay is already in retrieval; surface it in the UI: *"Cited document is 1,431 days old"* in red next to the inline evidence pill. Sometimes old evidence is correct (history of MI), sometimes it's misleading (last A1c). Surfacing the *age* lets the clinician decide.

### 4.6 Realistic patient demographics layer (S)
Use a small fixed mapping `subject_id → {fake_name, age, sex, demographic_tags}` so the demo doesn't constantly say "subject 13221453." The MIMIC dataset already gives you DOB/sex; layer a name on top with [Faker](https://faker.readthedocs.io/) seeded by `subject_id`. (Document clearly: data is real-de-identified MIMIC, names are synthetic.)

### 4.7 "Compare to current EHR note" diff view (M)
Have a fake "current EHR draft" baseline (literally the latest discharge summary in the record) and show a side-by-side **diff** vs the Co-Pilot-generated SOAP note: green = added, red = removed, gray = same. This *visualizes* the value-add — the green text is what the AI surfaced that the existing chart doesn't have. Most "AI helps doctor" demos can't show this because they have nothing to compare against; you can.

### 4.8 Triage / urgency score on the patient list (S)
When you add multi-patient support to the sidebar, the patient list should be **ordered by acuity**, not by `subject_id`. A small per-patient pre-compute step: "this patient has a stale troponin and a new chest-pain mention → urgency 8/10". The clinician's morning starts with *"who do I see first?"* — a copilot that answers that question is doing something the EHR doesn't.

### 4.9 Scribe mode vs Pre-charting mode toggle (S)
Two real clinical workflows are wildly different:
- **Pre-charting** (before clinic): summary, questions to ask, differential.
- **Scribe** (during visit): live transcription, conversation summary, real-time alerts, draft SOAP.

Currently both are in the same screen. A top-level mode toggle (`Pre-Visit` / `Visit` / `Post-Visit`) that hides irrelevant panels would communicate that you understand the workflow phases — and would also clean up the layout for free.

---

## 5. Suggested top-5 to actually do

If you only have a weekend left before submission, in priority order:

1. **Demo narrative + patient header strip + Faker names** (1.1, 3.1, 4.6) — half a day, transforms the *feel* of the entire app.
2. **Differential diagnosis button** (2.1) — half a day, single most credible new clinical capability.
3. **Pre-charting "questions to ask" panel** (2.3) — 2 hours, the single feature real clinicians most want.
4. **FHIR output shape + "Send to EHR" mock button** (4.1) — half a day, signals deployability.
5. **Streaming summary + node-by-node status line** (3.5) — 2 hours, makes the existing flow feel 10× more responsive without changing the architecture.

Everything else on this list compounds — but those five together push the submission from "well-built RAG demo" to "plausibly a healthcare product."
