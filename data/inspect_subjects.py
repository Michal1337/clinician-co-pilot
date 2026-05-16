#!/usr/bin/env python
"""Cross-reference candidate MIMIC-IV subject IDs against admissions /
diagnoses / procedures / meds / discharge notes so you can pick the
richest patient for the demo.

Use after a CXR-count run has narrowed the field to ~10-20 subjects.

Example:
    python inspect_subjects.py 13475033 19182863 15131736 14851532 \\
                               10933609 14841168 17340686 16826047 \\
                               --data-path /path/to/mimiciv/3.1 \\
                               --notes-path /path/to/mimic-iv-note/2.2

Outputs a single table sorted by a composite score (admissions × note
content × diagnosis breadth), so the first row is the best demo
candidate. Skips lab counting on purpose — labs.csv.gz is ~50 GB and the
demo doesn't depend on lab volume per se."""

import argparse
import os
from datetime import datetime

import pandas as pd

DEFAULT_DATA_PATH = os.environ.get(
    "MIMICIV_PATH",
    "/mnt/evafs/groups/re-com/mgromadzki/physionet.org/files/mimiciv/3.1",
)
DEFAULT_NOTES_PATH = os.environ.get(
    "MIMICIV_NOTES_PATH",
    "/mnt/evafs/groups/re-com/mgromadzki/physionet.org/files/mimic-iv-note/2.2",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("subjects", nargs="+", type=int, help="Subject IDs to inspect")
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--notes-path", default=DEFAULT_NOTES_PATH)
    p.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only print the top N rows (by composite score).",
    )
    return p.parse_args()


def _safe_count(df: pd.DataFrame, sids: set, col: str = "subject_id") -> pd.Series:
    return df[df[col].isin(sids)].groupby(col).size()


def _year_span(g: pd.DataFrame) -> float:
    try:
        ts = pd.to_datetime(g["admittime"], errors="coerce").dropna()
        if len(ts) < 2:
            return 0.0
        return round((ts.max() - ts.min()).days / 365.25, 1)
    except Exception:
        return 0.0


def main():
    args = parse_args()
    sids = set(args.subjects)
    print(f"Inspecting {len(sids)} subjects from {args.data_path}")

    print("  loading patients…")
    pats = pd.read_csv(
        f"{args.data_path}/hosp/patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age"],
    )
    pats = pats[pats.subject_id.isin(sids)].set_index("subject_id")

    print("  loading admissions…")
    adm = pd.read_csv(
        f"{args.data_path}/hosp/admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime"],
    )
    adm = adm[adm.subject_id.isin(sids)]
    adm_counts = adm.groupby("subject_id").size()
    span_yrs = adm.groupby("subject_id").apply(_year_span)

    print("  loading diagnoses…")
    dx = pd.read_csv(
        f"{args.data_path}/hosp/diagnoses_icd.csv.gz",
        usecols=["subject_id", "icd_code"],
    )
    dx_filtered = dx[dx.subject_id.isin(sids)]
    dx_counts = dx_filtered.groupby("subject_id").size()
    dx_unique = dx_filtered.groupby("subject_id")["icd_code"].nunique()

    print("  loading procedures…")
    proc = pd.read_csv(
        f"{args.data_path}/hosp/procedures_icd.csv.gz", usecols=["subject_id"]
    )
    proc_counts = _safe_count(proc, sids)

    print("  loading prescriptions…")
    rx = pd.read_csv(
        f"{args.data_path}/hosp/prescriptions.csv.gz",
        usecols=["subject_id"],
        low_memory=False,
    )
    rx_counts = _safe_count(rx, sids)

    print("  loading discharge notes…")
    notes = pd.read_csv(
        f"{args.notes_path}/note/discharge.csv.gz",
        usecols=["subject_id", "text"],
    )
    notes_f = notes[notes.subject_id.isin(sids)]
    note_counts = notes_f.groupby("subject_id").size()
    note_kchars = notes_f.groupby("subject_id")["text"].apply(
        lambda s: int(s.str.len().sum() / 1000)
    )

    rows = []
    for sid in args.subjects:
        rows.append(
            {
                "subject_id": sid,
                "age": int(pats.loc[sid, "anchor_age"]) if sid in pats.index else None,
                "sex": pats.loc[sid, "gender"] if sid in pats.index else None,
                "adm": int(adm_counts.get(sid, 0)),
                "yrs": float(span_yrs.get(sid, 0.0) or 0.0),
                "dx": int(dx_counts.get(sid, 0)),
                "dx_unique": int(dx_unique.get(sid, 0)),
                "proc": int(proc_counts.get(sid, 0)),
                "rx": int(rx_counts.get(sid, 0)),
                "notes": int(note_counts.get(sid, 0)),
                "note_kchars": int(note_kchars.get(sid, 0)),
            }
        )

    df = pd.DataFrame(rows)
    # Composite score: heavy on note content + diagnostic breadth, plus
    # admissions and a small bonus for longitudinal span (>=1y).
    df["score"] = (
        df["note_kchars"] * 1.0
        + df["dx_unique"] * 2.0
        + df["adm"] * 10.0
        + df["yrs"].clip(upper=10) * 5.0
    )
    df = df.sort_values("score", ascending=False)
    if args.top:
        df = df.head(args.top)

    print()
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 0)
    print(df.to_string(index=False))
    print()
    print(
        "Pick: high `adm` + high `note_kchars` + reasonable `yrs` span "
        "(longitudinal story) + moderate `dx_unique` (rich but not 100+ "
        "ICD codes — avoid super-complex ICU cases that overwhelm the UI)."
    )


if __name__ == "__main__":
    main()
