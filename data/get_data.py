#!/usr/bin/env python
"""Extract per-patient histories from MIMIC-IV (PhysioNet) and the
MIMIC-CXR-RRG findings dataset.

Output: one ``patient_<subject_id>_history.json`` per patient in the current
directory plus an ``cxrs/`` folder of saved chest-X-ray JPEGs. The files
are then consumed by ``../src/make_vdbs.py``.

Usage:
    python get_data.py --subjects 13221453 18765432
    python get_data.py --subjects-file subjects.txt
    # or fall back to the default demo patient if no flag is given.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

DEFAULT_DATA_PATH = (
    "/mnt/evafs/groups/re-com/mgromadzki/physionet.org/files/mimiciv/3.1"
)
DEFAULT_NOTES_PATH = (
    "/mnt/evafs/groups/re-com/mgromadzki/physionet.org/files/mimic-iv-note/2.2"
)
DEFAULT_DEMO_SUBJECTS = [13221453]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-path",
        default=os.environ.get("MIMICIV_PATH", DEFAULT_DATA_PATH),
        help="Path to the unpacked physionet.org/files/mimiciv/3.1 directory.",
    )
    p.add_argument(
        "--notes-path",
        default=os.environ.get("MIMICIV_NOTES_PATH", DEFAULT_NOTES_PATH),
        help="Path to the unpacked physionet.org/files/mimic-iv-note/2.2 directory.",
    )
    p.add_argument(
        "--cxr-dataset",
        default=os.environ.get("MIMIC_CXR_DATASET", "Yamini-1628/MIMIC-CXR-RRG"),
        help="HuggingFace dataset (or local path) for chest X-ray findings.",
    )
    p.add_argument(
        "--cxr-split", default="test", help="Dataset split that contains imaging."
    )
    p.add_argument(
        "--subjects",
        nargs="*",
        type=int,
        default=None,
        help="Subject IDs to extract. Can be combined with --subjects-file.",
    )
    p.add_argument(
        "--subjects-file",
        type=Path,
        default=None,
        help="Newline-separated file of subject IDs.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where patient_*.json and cxrs/ are written.",
    )
    p.add_argument(
        "--target-date",
        default="2025-12-01",
        help="MIMIC anonymizes by shifting every patient's timeline forward "
             "by a different ~100-year offset, so a single global shift only "
             "works for one patient. Instead, we compute the shift "
             "PER-PATIENT so each patient's most recent admission lands "
             "near this target date (YYYY-MM-DD). Picks ~6 months before "
             "'today' by default so the chart reads as a real recent record.",
    )
    return p.parse_args()


def collect_subject_ids(args):
    ids = list(args.subjects or [])
    if args.subjects_file and args.subjects_file.exists():
        with open(args.subjects_file) as f:
            ids.extend(int(line.strip()) for line in f if line.strip())
    if not ids:
        print(
            f"No --subjects supplied; falling back to demo subject {DEFAULT_DEMO_SUBJECTS}."
        )
        ids = list(DEFAULT_DEMO_SUBJECTS)
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for s in ids:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def load_tables(data_path: str, notes_path: str):
    print(f"Loading MIMIC-IV from {data_path}")
    return {
        "patients": pd.read_csv(f"{data_path}/hosp/patients.csv.gz"),
        "admissions": pd.read_csv(f"{data_path}/hosp/admissions.csv.gz"),
        "diagnoses": pd.read_csv(f"{data_path}/hosp/diagnoses_icd.csv.gz"),
        "d_icd_diag": pd.read_csv(f"{data_path}/hosp/d_icd_diagnoses.csv.gz"),
        "procedures": pd.read_csv(f"{data_path}/hosp/procedures_icd.csv.gz"),
        "d_icd_proc": pd.read_csv(f"{data_path}/hosp/d_icd_procedures.csv.gz"),
        "prescriptions": pd.read_csv(
            f"{data_path}/hosp/prescriptions.csv.gz", low_memory=False
        ),
        "labevents": pd.read_csv(f"{data_path}/hosp/labevents.csv.gz"),
        "d_labitems": pd.read_csv(f"{data_path}/hosp/d_labitems.csv.gz"),
        "discharge_notes": pd.read_csv(f"{notes_path}/note/discharge.csv.gz"),
    }


def _compute_patient_shift(patient_adm, target_date_str: str):
    """MIMIC anonymizes by shifting every patient's timeline forward by a
    different (typically 100+-year) offset. Pick a per-patient shift so
    the patient's latest admission discharge lands near ``target_date``.
    Returns a ``relativedelta`` to add to every date for that patient."""
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    candidates = []
    for col in ("dischtime", "admittime"):
        if col not in patient_adm.columns:
            continue
        for s in patient_adm[col].dropna():
            try:
                candidates.append(datetime.strptime(str(s), "%Y-%m-%d %H:%M:%S"))
            except (ValueError, TypeError):
                continue
    if not candidates:
        return relativedelta(years=0)
    return relativedelta(target_dt, max(candidates))


def build_patient_history(subject_id: int, t: dict, args):
    pat_row = t["patients"][t["patients"].subject_id == subject_id]
    if pat_row.empty:
        raise ValueError(f"Subject {subject_id} not found in patients table.")
    patient = pat_row.iloc[0]
    patient_adm = t["admissions"][t["admissions"].subject_id == subject_id]
    shift = _compute_patient_shift(patient_adm, args.target_date)

    history = {
        "subject_id": subject_id,
        "demographics": {
            "sex": patient["gender"],
            "anchor_age": int(patient["anchor_age"]),
        },
        "admissions": [],
    }

    for _, adm in tqdm(
        patient_adm.iterrows(),
        total=len(patient_adm),
        desc=f"subject {subject_id}",
    ):
        hadm_id = adm["hadm_id"]

        dx = t["diagnoses"][t["diagnoses"].hadm_id == hadm_id]
        dx = dx.merge(t["d_icd_diag"], on=["icd_code", "icd_version"], how="left")
        dx_list = dx["long_title"].dropna().unique().tolist()

        proc = t["procedures"][t["procedures"].hadm_id == hadm_id]
        proc = proc.merge(t["d_icd_proc"], on=["icd_code", "icd_version"], how="left")
        proc_list = proc["long_title"].dropna().unique().tolist()

        meds = t["prescriptions"][t["prescriptions"].hadm_id == hadm_id]
        med_list = meds["drug"].dropna().unique().tolist()

        labs = t["labevents"][t["labevents"].hadm_id == hadm_id]
        labs = labs.merge(t["d_labitems"], on="itemid", how="left")
        lab_summary = {}
        for labname, group in labs.groupby("label"):
            if len(lab_summary) > 10:
                break
            numeric = pd.to_numeric(group["valuenum"], errors="coerce").dropna()
            if len(numeric) > 0:
                lab_summary[labname] = {
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                }

        note = t["discharge_notes"][t["discharge_notes"].hadm_id == hadm_id]
        discharge_text = note.iloc[0]["text"] if len(note) > 0 else ""

        adm_dt = datetime.strptime(adm["admittime"], "%Y-%m-%d %H:%M:%S")
        dis_dt = datetime.strptime(adm["dischtime"], "%Y-%m-%d %H:%M:%S")

        history["admissions"].append(
            {
                "hadm_id": int(hadm_id),
                "admittime": str(adm_dt + shift),
                "dischtime": str(dis_dt + shift),
                "diagnoses": dx_list,
                "procedures": proc_list,
                "medications": med_list,
                "lab_summary": lab_summary,
                "discharge_summary": discharge_text,
            }
        )

    return history, shift


def _shift_study_date(raw, shift) -> str:
    """MIMIC-CXR ``StudyDate`` is a date-shifted YYYYMMDD (often as int or
    string). Apply the same shift used for admissions so the timeline lines
    up. Returns ISO YYYY-MM-DD on success, the original value on failure."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if len(s) == 8 and s.isdigit():
        try:
            dt = datetime.strptime(s, "%Y%m%d")
            return (dt + shift).strftime("%Y-%m-%d")
        except ValueError:
            return s
    # Already ISO-ish? Try to parse and shift.
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            return (dt + shift).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s


def attach_xrays(history: dict, cxr_dataset, output_dir: Path, shift):
    sid = str(history["subject_id"])
    cxr_list = cxr_dataset.filter(lambda r, sid=sid: r["subject_id"] == sid)

    cxrs_dir = output_dir / "cxrs"
    cxrs_dir.mkdir(parents=True, exist_ok=True)

    cxrs = []
    for r in cxr_list:
        image_filename = cxrs_dir / f"{r['dicom_id']}.jpg"
        try:
            r["main_image"].save(image_filename)
        except Exception as e:
            print(f"Warning: could not save {image_filename}: {e}")
            continue
        cxrs.append(
            {
                "study_id": r.get("study_id"),
                "dicom_id": r.get("dicom_id"),
                "findings": r.get("findings_section"),
                "impression": r.get("impression_section"),
                "main_image_path": str(image_filename),
                "study_date": _shift_study_date(r.get("StudyDate"), shift),
                "study_time": r.get("StudyTime"),
            }
        )
    history["xray_studies"] = cxrs
    return history


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_ids = collect_subject_ids(args)

    tables = load_tables(args.data_path, args.notes_path)

    print(f"Loading CXR dataset {args.cxr_dataset} (split={args.cxr_split})")
    cxr_dataset = load_dataset(args.cxr_dataset, "findings_section", split=args.cxr_split)

    for sid in subject_ids:
        try:
            history, shift = build_patient_history(sid, tables, args)
        except ValueError as e:
            print(f"Skipping {sid}: {e}")
            continue
        history = attach_xrays(history, cxr_dataset, args.output_dir, shift)
        out_path = args.output_dir / f"patient_{sid}_history.json"
        with open(out_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Wrote {out_path} ({len(history['admissions'])} admissions, "
              f"{len(history['xray_studies'])} CXR studies, "
              f"shift={shift.years}y{shift.months:+d}m -> target {args.target_date})")


if __name__ == "__main__":
    main()
