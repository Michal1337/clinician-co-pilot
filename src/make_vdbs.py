import json
import uuid

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import IMAGE_EMBEDDINGS, TEXT_EMBEDDINGS

JSON_PATH = "../data/patient_13221453_history.json"

with open(JSON_PATH, "r") as f:
    history = json.load(f)

subject_id = history["subject_id"]

# Text splitter ONLY for discharge summaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=128, chunk_overlap=32, length_function=lambda x: len(x.split())
)

img_documents = []
documents = []

for adm in history["admissions"]:

    base_metadata = {
        "subject_id": subject_id,
        "hadm_id": adm["hadm_id"],
        "admittime": adm["admittime"],
        "dischtime": adm["dischtime"],
    }

    # ---- Diagnoses ----
    if adm["diagnoses"]:
        text = "Diagnosis list:\n" + "\n".join(adm["diagnoses"])
        documents.append(
            Document(
                page_content=text,
                metadata={
                    **base_metadata,
                    "section": "diagnoses",
                    "document_id": str(uuid.uuid4()),
                },
            )
        )

    # ---- Procedures ----
    if adm["procedures"]:
        text = "Procedure list:\n" + "\n".join(adm["procedures"])
        documents.append(
            Document(
                page_content=text,
                metadata={
                    **base_metadata,
                    "section": "procedures",
                    "document_id": str(uuid.uuid4()),
                },
            )
        )

    # ---- Medications ----
    if adm["medications"]:
        text = "Medication list:\n" + "\n".join(adm["medications"])
        documents.append(
            Document(
                page_content=text,
                metadata={
                    **base_metadata,
                    "section": "medications",
                    "document_id": str(uuid.uuid4()),
                },
            )
        )

    # ---- Lab Summary ----
    if adm["lab_summary"]:
        lab_lines = [
            f"{lab}: min={vals['min']} max={vals['max']}"
            for lab, vals in adm["lab_summary"].items()
        ]
        text = "Laboratory summary:\n" + "\n".join(lab_lines)

        documents.append(
            Document(
                page_content=text,
                metadata={
                    **base_metadata,
                    "section": "lab_summary",
                    "document_id": str(uuid.uuid4()),
                },
            )
        )

    # ---- Discharge Summary (chunked) ----
    discharge_text = adm.get("discharge_summary", "")
    if discharge_text:
        chunks = splitter.split_text(discharge_text)

        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **base_metadata,
                        "section": "discharge_summary",
                        "chunk_id": i,
                        "document_id": str(uuid.uuid4()),
                    },
                )
            )

print(f"Created {len(documents)} LangChain documents.")

for study in history["xray_studies"]:
    img_documents.append(
        Document(
            page_content=study["main_image_path"],
            metadata={
                "subject_id": subject_id,
                "study_id": study["study_id"],
                "dicom_id": study["dicom_id"],
                "findings": study.get("findings"),
                "impression": study.get("impression"),
                "study_date": study.get("study_date"),
                "study_time": study.get("study_time"),
            },
        )
    )

print(f"Created {len(img_documents)} Image LangChain documents.")

text_vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=TEXT_EMBEDDINGS,
)

text_vectorstore.save_local("../data/vdbs/text_vdb")

img_vectorstore = FAISS.from_documents(
    documents=img_documents,
    embedding=IMAGE_EMBEDDINGS,
)

img_vectorstore.save_local("../data/vdbs/img_vdb")
