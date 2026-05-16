import glob
import json
import os
import uuid
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import IMAGE_EMBEDDINGS, TEXT_EMBEDDINGS

DATA_DIR = Path("../data")
HISTORY_GLOB = str(DATA_DIR / "patient_*_history.json")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=384, chunk_overlap=96, length_function=lambda x: len(x.split())
)


def _normalize_image_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p) or os.path.exists(p):
        return p
    candidate = DATA_DIR / p
    if candidate.exists():
        return str(candidate)
    return p


def documents_for_patient(history: dict):
    subject_id = history["subject_id"]
    text_docs = []
    img_docs = []

    for adm in history.get("admissions", []):
        base_metadata = {
            "subject_id": subject_id,
            "hadm_id": adm["hadm_id"],
            "admittime": adm["admittime"],
            "dischtime": adm["dischtime"],
        }

        if adm.get("diagnoses"):
            text = "Diagnosis list:\n" + "\n".join(adm["diagnoses"])
            text_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        **base_metadata,
                        "section": "diagnoses",
                        "document_id": str(uuid.uuid4()),
                    },
                )
            )

        if adm.get("procedures"):
            text = "Procedure list:\n" + "\n".join(adm["procedures"])
            text_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        **base_metadata,
                        "section": "procedures",
                        "document_id": str(uuid.uuid4()),
                    },
                )
            )

        if adm.get("medications"):
            text = "Medication list:\n" + "\n".join(adm["medications"])
            text_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        **base_metadata,
                        "section": "medications",
                        "document_id": str(uuid.uuid4()),
                    },
                )
            )

        if adm.get("lab_summary"):
            lab_lines = [
                f"{lab}: min={vals['min']} max={vals['max']}"
                for lab, vals in adm["lab_summary"].items()
            ]
            text = "Laboratory summary:\n" + "\n".join(lab_lines)
            text_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        **base_metadata,
                        "section": "lab_summary",
                        "document_id": str(uuid.uuid4()),
                    },
                )
            )

        discharge_text = adm.get("discharge_summary", "")
        if discharge_text:
            for i, chunk in enumerate(splitter.split_text(discharge_text)):
                text_docs.append(
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

    for study in history.get("xray_studies", []):
        path = _normalize_image_path(study.get("main_image_path", ""))
        img_docs.append(
            Document(
                page_content=path,
                metadata={
                    "subject_id": subject_id,
                    "study_id": study.get("study_id"),
                    "dicom_id": study.get("dicom_id"),
                    "findings": study.get("findings"),
                    "impression": study.get("impression"),
                    "study_date": study.get("study_date"),
                    "study_time": study.get("study_time"),
                    "main_image_path": path,
                },
            )
        )

    return text_docs, img_docs


def main():
    history_paths = sorted(glob.glob(HISTORY_GLOB))
    if not history_paths:
        raise SystemExit(
            f"No patient history files found at {HISTORY_GLOB}. "
            "Run data/get_data.py first."
        )

    all_text_docs = []
    all_img_docs = []
    subject_ids = []
    for hp in history_paths:
        with open(hp, "r") as f:
            history = json.load(f)
        text_docs, img_docs = documents_for_patient(history)
        all_text_docs.extend(text_docs)
        all_img_docs.extend(img_docs)
        subject_ids.append(history["subject_id"])
        print(
            f"[{hp}] subject_id={history['subject_id']}: "
            f"{len(text_docs)} text docs, {len(img_docs)} image docs"
        )

    print(
        f"Total: {len(all_text_docs)} text docs, {len(all_img_docs)} image docs "
        f"across {len(subject_ids)} patient(s)."
    )

    text_vectorstore = FAISS.from_documents(
        documents=all_text_docs, embedding=TEXT_EMBEDDINGS
    )
    text_vectorstore.save_local(str(DATA_DIR / "vdbs" / "text_vdb"))

    if all_img_docs:
        img_vectorstore = FAISS.from_documents(
            documents=all_img_docs, embedding=IMAGE_EMBEDDINGS
        )
        img_vectorstore.save_local(str(DATA_DIR / "vdbs" / "img_vdb"))
    else:
        print("No imaging studies found; skipping image vector store.")

    # Persist the patient roster so the app can offer a selectbox without
    # re-loading every JSON.
    roster_path = DATA_DIR / "vdbs" / "patients.json"
    roster_path.parent.mkdir(parents=True, exist_ok=True)
    with open(roster_path, "w") as f:
        json.dump(sorted(set(subject_ids)), f)
    print(f"Wrote patient roster -> {roster_path}")


if __name__ == "__main__":
    main()
