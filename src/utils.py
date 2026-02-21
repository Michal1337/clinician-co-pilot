import math
import re
from datetime import datetime

import streamlit as st


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
    res = "\n".join(f"{item2str[item]}: {doc.metadata[item]}" for item in ["document_id", "hadm_id", "admittime", "dischtime", "section"])
    res += "\nDocument Content:" + doc.page_content
    return res

def imagedoc2str(doc):
    res = "\n".join(f"{item2str[item]}: {doc.metadata[item]}" for item in ["doc_id", "date", "impression", "findings"])
    return res

def windowed_time_decay(doc_date_str, allowed_years, lambda_inside=0.005, lambda_outside=0.03):
    doc_date = datetime.strptime(doc_date_str, "%Y-%m-%d %H:%M:%S")
    now = datetime.now()
    age_days = (now - doc_date).days
    cutoff_days = allowed_years * 365
    if age_days <= cutoff_days:
        return math.exp(-lambda_inside * age_days)
    else:
        return math.exp(-lambda_outside * age_days)

def normalize_medasr(text: str) -> str:
    # 1. Normalize special tokens
    replacements = {
        "{period}": ".",
        "{comma}": ",",
        "{colon}": ":",
        "{new paragraph}": "\n\n",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # 2. Fix glued section headers like "TECchHNIQe[TECHNIQUE]"
    text = re.sub(r"[A-Za-z]+\[([A-Z ]+)\]", r"[\1]", text)

    # 3. Remove accidental duplicated fragments like:
    # segmental.segmental → segmental
    # pneumothorax.othorax → pneumothorax
    text = re.sub(r"\b(\w+)\.\1\b", r"\1", text)

    # 4. Fix partial word repetition after period (pneumothorax.othorax)
    text = re.sub(r"\b(\w+)\.(\w+)\b", 
                  lambda m: m.group(1) if m.group(2) in m.group(1) else m.group(0),
                  text)

    # 5. Add newline before section headers
    text = re.sub(r"\s*(\[[A-Z ]+\])", r"\n\1", text)

    # 6. Normalize spacing
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)

    # 7. Clean punctuation spacing
    text = re.sub(r"\s+([.,:])", r"\1", text)
    text = re.sub(r"([.,:])(?=[^\s])", r"\1 ", text)

    return text.strip()

def render_stage1(state, query_action_box, summary_box):
    """Render current action/query and a patient summary, skipping empty sections.
    Evidence is rendered with the source id as inline code and the date in italics.
    Handles both string and dict items safely.
    """
    # Show current action/query
    if state.get("action") != "finish":
        query_action_box.markdown(
            f"### 🔎 Current State\n**Action:** `{state.get('action')}`  \n**Query:** `{state.get('query')}`  \n**Allowed Years:** `{state.get('allowed_years')}`"
        )
    else:
        query_action_box = st.empty()

    summary = state.get("summary") or {}
    if not summary:
        summary_box.markdown("No summary yet.")
        return

    def nonempty_str(s):
        return isinstance(s, str) and s.strip() != ""

    def get_list(obj, key):
        val = obj.get(key)
        if val is None:
            return []
        if isinstance(val, list):
            return val
        # If accidentally a string, wrap in list
        if isinstance(val, str):
            return [val]
        return []

    def item_has_any_text(item, keys):
        if isinstance(item, str):
            return nonempty_str(item)
        # dict case
        for k in keys:
            if nonempty_str(item.get(k, "")):
                return True
        for ev in get_list(item, "evidence"):
            if nonempty_str(ev.get("source_id", "")) or nonempty_str(ev.get("date", "")):
                return True
        return False

    def render_evidence(evs, out_lines):
        for ev in evs:
            src = (ev.get("source_id") or "").strip()
            date = (ev.get("date") or "").strip()
            if not (src or date):
                continue
            if src and date:
                out_lines.append(f"  - Evidence: (`{src}`) _({date})_")
            elif src:
                out_lines.append(f"  - Evidence: (`{src}`)")
            else:
                out_lines.append(f"  - Evidence: _({date})_")

    md_lines = ["### 🧾 Patient Summary\n"]
    added_content = False

    # Helper to render sections
    def render_section(title, items, keys, formatter=None):
        nonlocal added_content
        filtered = [i for i in items if item_has_any_text(i, keys)]
        if not filtered:
            return
        added_content = True
        md_lines.append(f"#### {title}")
        for i in filtered:
            if isinstance(i, str):
                md_lines.append(f"- {i}")
            else:
                line = formatter(i) if formatter else str(i)
                md_lines.append(line)
        md_lines.append("")

    # Active Problems
    def format_problem(p):
        prob = (p.get("problem") or "").strip() or "(unspecified problem)"
        status = (p.get("status") or "").strip()
        line = f"- **{prob}**" + (f" — {status}" if status else "")
        render_evidence(get_list(p, "evidence"), md_lines)
        return line

    render_section("⚠️ Active Problems", get_list(summary, "active_problems"), ["problem", "status"], format_problem)

    # Recent Events
    def format_event(e):
        event_text = (e.get("event") or "").strip()
        date = (e.get("date") or "").strip()
        return f"- {event_text}" + (f" — {date}" if date else "")

    render_section("📝 Recent Events", get_list(summary, "recent_events"), ["event"], format_event)

    # Medications
    def format_med(m):
        name = (m.get("name") or "").strip()
        dose = (m.get("dose") or "").strip()
        route = (m.get("route") or "").strip()
        parts = [p for p in [name, (f"dose: {dose}" if dose else ""), (f"route: {route}" if route else "")] if p]
        line = f"- {' — '.join(parts)}"
        render_evidence(get_list(m, "evidence"), md_lines)
        return line

    render_section("💊 Medications", get_list(summary, "medications"), ["name", "dose", "route"], format_med)

    # Key Results
    def format_result(r):
        test = (r.get("test") or "").strip() or "(test)"
        result = (r.get("result") or "").strip()
        date = (r.get("date") or "").strip()
        line = f"- **{test}**"
        if result:
            line += f": {result}"
        if date:
            line += f" — {date}"
        render_evidence(get_list(r, "evidence"), md_lines)
        return line

    render_section("📊 Key Results", get_list(summary, "key_results"), ["test", "result"], format_result)

    # Procedures
    def format_proc(p):
        proc = (p.get("procedure") or "").strip()
        date = (p.get("date") or "").strip()
        line = f"- {proc}" + (f" — {date}" if date else "")
        render_evidence(get_list(p, "evidence"), md_lines)
        return line

    render_section("🏥 Procedures", get_list(summary, "procedures"), ["procedure"], format_proc)

    # Allergies
    def format_allergy(a):
        sub = (a.get("substance") or "").strip()
        reaction = (a.get("reaction") or "").strip()
        if sub and reaction:
            line = f"- {sub} — {reaction}"
        elif sub:
            line = f"- {sub}"
        elif reaction:
            line = f"- (reaction) {reaction}"
        else:
            line = "- (unspecified)"
        render_evidence(get_list(a, "evidence"), md_lines)
        return line

    render_section("⚠️ Allergies", get_list(summary, "allergies"), ["substance", "reaction"], format_allergy)

    # Pending Items
    def format_pending(p):
        item = (p.get("item") or "").strip()
        line = f"- {item}" if item else "- (unspecified)"
        render_evidence(get_list(p, "evidence"), md_lines)
        return line

    render_section("⏳ Pending Items", get_list(summary, "pending_items"), ["item"], format_pending)

    if not added_content:
        summary_box.markdown("### 🧾 Patient Summary\nNo populated sections in the summary yet.")
        return

    summary_box.markdown("\n".join(md_lines))