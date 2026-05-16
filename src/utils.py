import json
import math
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from json_repair import repair_json


# Gemma 4 wraps its optional thought channel: `<channel|>thought<channel|>answer`.
GEMMA4_THINK_SEP = "<channel|>"
LEGACY_THINK_SEP = "<unused95>"

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _assistant_content(response_or_text: Any) -> str:
    if isinstance(response_or_text, str):
        return response_or_text
    if isinstance(response_or_text, list):
        try:
            content = response_or_text[0]["generated_text"][-1]["content"]
        except (KeyError, IndexError, TypeError):
            return str(response_or_text)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")
        return str(content)
    return str(response_or_text)


def extract_assistant_text(response_or_text: Any) -> str:
    text = _assistant_content(response_or_text)
    for sep in (GEMMA4_THINK_SEP, LEGACY_THINK_SEP):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
            break
    return text.strip()


def parse_response_json(response_or_text: Any) -> Any:
    text = extract_assistant_text(response_or_text)
    text = _CODE_FENCE_RE.sub("", text).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(repair_json(text))
        except Exception:
            return {}


def extract_response_json(text: Any) -> str:
    cleaned = extract_assistant_text(text)
    return _CODE_FENCE_RE.sub("", cleaned).strip()


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


# --- Image-path resolution -----------------------------------------------

@lru_cache(maxsize=4096)
def _resolve_path_cached(candidate: str) -> Optional[str]:
    if not candidate:
        return None
    if os.path.isabs(candidate):
        return candidate if os.path.exists(candidate) else None
    project_root = Path(__file__).resolve().parent.parent
    for root in (Path.cwd(), project_root, project_root / "data"):
        p = (root / candidate).resolve()
        if p.exists():
            return str(p)
    return None


def resolve_image_path(doc) -> Optional[str]:
    meta = getattr(doc, "metadata", None) or {}
    for key in ("main_image_path", "image_path", "path", "file_path"):
        v = meta.get(key)
        if isinstance(v, str) and v:
            resolved = _resolve_path_cached(v)
            if resolved:
                return resolved
    return None


def collect_image_paths(retrieved_docs_batches, max_images: int = 6) -> List[str]:
    seen = set()
    out: List[str] = []
    for batch in retrieved_docs_batches or []:
        for doc in batch or []:
            path = resolve_image_path(doc)
            if path and path not in seen:
                seen.add(path)
                out.append(path)
                if len(out) >= max_images:
                    return out
    return out


# --- Source-id index for friendly evidence rendering ---------------------

def _normalize_section(section: Optional[str]) -> str:
    if not section:
        return "Source"
    label = section.replace("_", " ").strip().title()
    overrides = {
        "Discharge Summary": "Discharge note",
        "Lab Summary": "Labs",
        "Diagnoses": "Diagnoses",
        "Procedures": "Procedures",
        "Medications": "Medications",
    }
    return overrides.get(label, label)


def build_source_index(*vectorstores) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for vs in vectorstores:
        store = getattr(vs, "docstore", None)
        if store is None:
            continue
        items = getattr(store, "_dict", None) or {}
        for doc in items.values():
            m = getattr(doc, "metadata", {}) or {}
            section = _normalize_section(m.get("section"))
            date = m.get("admittime") or m.get("dischtime") or m.get("study_date")
            entry = {
                "section": section if m.get("section") else None,
                "date": date,
                "hadm_id": m.get("hadm_id"),
                "subject_id": m.get("subject_id"),
            }
            for key in ("document_id", "dicom_id", "study_id"):
                src = m.get(key)
                if src:
                    if key in ("dicom_id", "study_id") and not entry["section"]:
                        entry = dict(entry, section="Chest X-ray")
                    idx[str(src)] = entry
    return idx


def format_source_label(
    source_id: Optional[str],
    fallback_date: Optional[str],
    source_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    src = (source_id or "").strip()
    date = (fallback_date or "").strip()
    info = source_index.get(src) if (source_index and src) else None
    if info:
        section = info.get("section") or "Source"
        d = (date or info.get("date") or "")[:10]
        return f"{section}" + (f" · {d}" if d else "")
    if src and date:
        return f"`{src[:8]}…` · {date[:10]}"
    if src:
        return f"`{src[:8]}…`"
    return date[:10] if date else ""


# --- Multimodal user content --------------------------------------------

def add_images_to_user_content(content, retrieved_docs, max_images=3):
    # Gemma 4 docs: image entries must come before text in user content.
    if not isinstance(content, list):
        return content

    image_entries = []
    seen_paths = set()

    for doc in retrieved_docs or []:
        path = resolve_image_path(doc)
        if path and path not in seen_paths:
            image_entries.append({"type": "image", "url": path})
            seen_paths.add(path)
        if len(image_entries) >= max_images:
            break

    return image_entries + content if image_entries else content


# --- Time decay ----------------------------------------------------------

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


# --- JSON patch ----------------------------------------------------------

# Per-section primary content field: if this is blank, the item is bogus
# even if it carries an evidence citation. Drops the empty-shell entries
# the planner sometimes emits when a retrieved doc only mentions a topic.
_PRIMARY_CONTENT_FIELD = {
    "active_problems": "problem",
    "medications": "name",
    "recent_events": "event",
    "allergies": "substance",
    "key_results": "test",
    "procedures": "procedure",
    "pending_items": "item",
}


def _has_primary_content(item: Any, section: str) -> bool:
    if not isinstance(item, dict):
        return True
    key = _PRIMARY_CONTENT_FIELD.get(section)
    if not key:
        return True
    v = item.get(key)
    return isinstance(v, str) and bool(v.strip())


def _has_evidence(item: Any) -> bool:
    if not isinstance(item, dict):
        return True
    evs = item.get("evidence")
    if not isinstance(evs, list) or not evs:
        return False
    for ev in evs:
        if not isinstance(ev, dict):
            continue
        src = (ev.get("source_id") or "").strip()
        date = (ev.get("date") or "").strip()
        if src or date:
            return True
    return False


def apply_patch(original: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    def is_empty_item(item: dict) -> bool:
        for v in item.values():
            if isinstance(v, list):
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

        original[section] = [
            item
            for item in original[section]
            if not is_empty_item(item)
            and _has_primary_content(item, section)
            and _has_evidence(item)
        ]

        for item in new_items:
            if not isinstance(item, dict) or is_empty_item(item):
                continue
            if not _has_primary_content(item, section):
                continue
            if not _has_evidence(item):
                continue
            if item not in original[section]:
                original[section].append(item)

    return original


# --- Planner guardrails --------------------------------------------------

_HISTORICAL_QUERY_HINTS = (
    "initial diagnos",
    "first diagnos",
    "past surgical",
    "history of surger",
    "lifetime",
    "genetic",
    "anchor age",
    "birth",
    "childhood",
)

# Fallback windows when the planner forgets to provide one, keyed by query substring.
_DEFAULT_WINDOW_BY_HINT = (
    ("medication", 2),
    ("med list", 2),
    ("lab", 1),
    ("vital", 1),
    ("hba1c", 1),
    ("imaging", 5),
    ("x-ray", 5),
    ("xray", 5),
    ("echo", 3),
    ("procedure", 5),
    ("admission", 3),
    ("hospital", 3),
)


def looks_historical(query: str) -> bool:
    q = (query or "").lower()
    return any(h in q for h in _HISTORICAL_QUERY_HINTS)


def default_allowed_years(query: str) -> int:
    q = (query or "").lower()
    for hint, years in _DEFAULT_WINDOW_BY_HINT:
        if hint in q:
            return years
    return 3


def validate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        return {"action": "finish", "query": None}
    action = plan.get("action")
    if action not in ("search_text", "search_imaging", "finish"):
        return {"action": "finish", "query": None}
    if action in ("search_text", "search_imaging"):
        query = plan.get("query") or ""
        years = plan.get("allowed_years")
        if (years is None or years == 0) and not looks_historical(query):
            plan["allowed_years"] = default_allowed_years(query)
    return plan
