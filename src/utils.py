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
    res = "\n".join(
        f"{item2str[item]}: {doc.metadata[item]}"
        for item in ["doc_id", "date", "impression", "findings"]
    )
    return res


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
