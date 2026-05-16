import json
import random
from pathlib import Path
from typing import Dict, Optional

DATA_DIR = Path("../data")

# Generic surname pool — MIMIC is de-identified, name is a UI affordance only.
_SURNAMES = [
    "Garcia", "Smith", "Nguyen", "Patel", "Cohen", "Okafor", "Johnson",
    "Martinez", "Rossi", "Chen", "Khan", "Anderson", "Schmidt", "Reyes",
    "Lopez", "Singh", "Brown", "Hernandez", "Park", "Williams", "Tanaka",
    "Diallo", "Andersen", "Davis", "Kowalski", "Ahmed", "Walker", "Ivanov",
]
_INITIALS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _synthesize_name(subject_id: int) -> str:
    rng = random.Random(int(subject_id))
    return f"{rng.choice(_SURNAMES)}, {rng.choice(_INITIALS)}."


def _format_mrn(subject_id: int) -> str:
    s = str(int(subject_id)).zfill(8)
    return f"MRN {s[:4]}-{s[4:]}"


def load_patient_demographics(subject_id: int) -> Dict[str, Optional[str]]:
    path = DATA_DIR / f"patient_{int(subject_id)}_history.json"
    age = None
    sex = None
    try:
        with open(path, "r") as f:
            history = json.load(f)
        demo = history.get("demographics") or {}
        age = demo.get("anchor_age")
        sex_raw = (demo.get("sex") or "").strip().upper()
        if sex_raw in ("M", "MALE"):
            sex = "Male"
        elif sex_raw in ("F", "FEMALE"):
            sex = "Female"
        elif sex_raw:
            sex = sex_raw.title()
    except (OSError, json.JSONDecodeError, KeyError):
        pass

    return {
        "subject_id": int(subject_id),
        "name": _synthesize_name(subject_id),
        "mrn": _format_mrn(subject_id),
        "age": age,
        "sex": sex,
    }


def primary_problem(summary: dict) -> Optional[str]:
    if not isinstance(summary, dict):
        return None
    for p in summary.get("active_problems") or []:
        if isinstance(p, dict):
            text = (p.get("problem") or "").strip()
        elif isinstance(p, str):
            text = p.strip()
        else:
            text = ""
        if text:
            return text
    return None


def allergy_badges(summary: dict, max_badges: int = 4) -> list:
    if not isinstance(summary, dict):
        return []
    out = []
    for a in summary.get("allergies") or []:
        if isinstance(a, dict):
            sub = (a.get("substance") or "").strip()
        elif isinstance(a, str):
            sub = a.strip()
        else:
            sub = ""
        if sub:
            out.append(sub)
        if len(out) >= max_badges:
            break
    return out
