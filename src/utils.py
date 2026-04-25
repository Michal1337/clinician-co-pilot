import os
import math
import re
from datetime import datetime
from typing import Any, Dict


def extract_response_json(text):
    text = text.replace("null", "None")
    try:
        clean = re.sub(r"^```json\s*|\s*```$", "", text.split("<unused95>")[1].strip())
    except:
        clean = re.sub(r"^```json\s*|\s*```$", "", text.strip())

    return clean


item2str = {
    "document_id": "Document ID",
    "hadm_id": "Admission ID",
    "admittime": "Admission Time",
    "dischtime": "Discharge Time",
    "section": "Document Type",
    "impression": "Xray Impression",
    "findings": "Xray Findings",
}


def textdoc2str(doc):
    res = "\n".join(
        f"{item2str[item]}: {doc.metadata[item]}"
        for item in ["document_id", "hadm_id", "admittime", "dischtime", "section"]
    )
    res += "\nDocument Content:" + doc.page_content
    return res


def imagedoc2str(doc):
    source_id = (
        doc.metadata.get("doc_id")
        or doc.metadata.get("document_id")
        or doc.metadata.get("dicom_id")
        or doc.metadata.get("study_id")
        or "N/A"
    )
    study_date = (
        doc.metadata.get("date")
        or doc.metadata.get("study_date")
        or doc.metadata.get("admittime")
        or "N/A"
    )
    impression = doc.metadata.get("impression") or "N/A"
    findings = doc.metadata.get("findings") or "N/A"
    res = (
        f"Document ID: {source_id}\n"
        f"Study Date: {study_date}\n"
        f"Xray Impression: {impression}\n"
        f"Xray Findings: {findings}"
    )
    return res


def add_images_to_user_content(content, retrieved_docs, max_images=3):
    """
    Add image entries to user message content for multimodal models.
    Images are inserted before text blocks as recommended by Gemma 4 docs.
    """
    if not isinstance(content, list):
        return content

    image_entries = []
    seen_paths = set()

    for doc in (retrieved_docs or []):
        path = None
        if isinstance(getattr(doc, "page_content", None), str) and os.path.exists(
            doc.page_content
        ):
            path = doc.page_content
        elif hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            for key in ["image_path", "main_image_path", "path", "file_path"]:
                maybe_path = doc.metadata.get(key)
                if isinstance(maybe_path, str) and os.path.exists(maybe_path):
                    path = maybe_path
                    break

        if path and path not in seen_paths:
            image_entries.append({"type": "image", "url": path})
            seen_paths.add(path)

        if len(image_entries) >= max_images:
            break

    return image_entries + content if image_entries else content


def windowed_time_decay(
    doc_date_str, allowed_years, lambda_inside=0.005, lambda_outside=0.03
):
    doc_date = datetime.strptime(doc_date_str, "%Y-%m-%d %H:%M:%S")
    now = datetime.now()
    age_days = (now - doc_date).days
    cutoff_days = allowed_years * 365
    if age_days <= cutoff_days:
        return math.exp(-lambda_inside * age_days)
    else:
        return math.exp(-lambda_outside * age_days)


def apply_patch(original: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    def is_empty_item(item: dict) -> bool:
        """Check if all values in the dict are empty or lists of empty dicts."""
        for v in item.values():
            if isinstance(v, list):
                # if list has at least one non-empty dict, keep it
                if any(not is_empty_item(x) for x in v if isinstance(x, dict)):
                    return False
            elif v not in ("", None):
                return False
        return True

    for section, new_items in patch.items():
        if section not in original or not isinstance(original[section], list):
            continue
        if not isinstance(new_items, list):
            continue

        # Remove any empty template entries first
        original[section] = [
            item for item in original[section] if not is_empty_item(item)
        ]

        # Append non-empty items, preventing duplicates
        for item in new_items:
            if not isinstance(item, dict) or is_empty_item(item):
                continue
            if item not in original[section]:
                original[section].append(item)

    return original
