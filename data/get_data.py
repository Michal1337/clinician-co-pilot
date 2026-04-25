#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datasets import load_dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
from tqdm import tqdm


# In[2]:


DATA_PATH = "/mnt/evafs/groups/re-com/mgromadzki/physionet.org/files/mimiciv/3.1"
DATA_PATH_NOTES = "/mnt/evafs/groups/re-com/mgromadzki/physionet.org/files/mimic-iv-note/2.2"


# In[3]:


patients = pd.read_csv(f"{DATA_PATH}/hosp/patients.csv.gz")
admissions = pd.read_csv(f"{DATA_PATH}/hosp/admissions.csv.gz")
diagnoses = pd.read_csv(f"{DATA_PATH}/hosp/diagnoses_icd.csv.gz")
d_icd_diag = pd.read_csv(f"{DATA_PATH}/hosp/d_icd_diagnoses.csv.gz")
procedures = pd.read_csv(f"{DATA_PATH}/hosp/procedures_icd.csv.gz")
d_icd_proc = pd.read_csv(f"{DATA_PATH}/hosp/d_icd_procedures.csv.gz")
prescriptions = pd.read_csv(f"{DATA_PATH}/hosp/prescriptions.csv.gz")
labevents = pd.read_csv(f"{DATA_PATH}/hosp/labevents.csv.gz")
d_labitems = pd.read_csv(f"{DATA_PATH}/hosp/d_labitems.csv.gz")

discharge_notes = pd.read_csv(f"{DATA_PATH_NOTES}/note/discharge.csv.gz")


# In[4]:


SUBJECT_ID = 13221453
patient = patients[patients.subject_id == SUBJECT_ID].iloc[0]
patient_adm = admissions[admissions.subject_id == SUBJECT_ID]

history = {
    "subject_id": SUBJECT_ID,
    "demographics": {
        "sex": patient["gender"],
        "anchor_age": int(patient["anchor_age"])
    },
    "admissions": []
}


# In[5]:


for _, adm in tqdm(patient_adm.iterrows()):
    hadm_id = adm["hadm_id"]

    # Diagnoses
    dx = diagnoses[diagnoses.hadm_id == hadm_id]
    dx = dx.merge(d_icd_diag, on=["icd_code", "icd_version"], how="left")
    dx_list = dx["long_title"].dropna().unique().tolist()

    # Procedures
    proc = procedures[procedures.hadm_id == hadm_id]
    proc = proc.merge(d_icd_proc, on=["icd_code", "icd_version"], how="left")
    proc_list = proc["long_title"].dropna().unique().tolist()

    # Medications
    meds = prescriptions[prescriptions.hadm_id == hadm_id]
    med_list = meds["drug"].dropna().unique().tolist()

    # Lab Summary (top 10 labs only to reduce size)
    labs = labevents[labevents.hadm_id == hadm_id]
    labs = labs.merge(d_labitems, on="itemid", how="left")

    lab_summary = {}
    for labname, group in labs.groupby("label"):
        if len(lab_summary) > 10:
            break
        numeric = pd.to_numeric(group["valuenum"], errors="coerce")
        numeric = numeric.dropna()
        if len(numeric) > 0:
            lab_summary[labname] = {
                "min": float(numeric.min()),
                "max": float(numeric.max())
            }

    # Discharge Summary
    note = discharge_notes[discharge_notes.hadm_id == hadm_id]
    discharge_text = ""
    if len(note) > 0:
        discharge_text = note.iloc[0]["text"]

    adm_dt = datetime.strptime(adm["admittime"], "%Y-%m-%d %H:%M:%S")
    dis_dt = datetime.strptime(adm["dischtime"], "%Y-%m-%d %H:%M:%S")

    admission_record = {
        "hadm_id": int(hadm_id),
        "admittime": str(adm_dt + relativedelta(years=-126, months=+8)),
        "dischtime": str(dis_dt + relativedelta(years=-126, months=+8)),
        "diagnoses": dx_list,
        "procedures": proc_list,
        "medications": med_list,
        "lab_summary": lab_summary,
        "discharge_summary": discharge_text
    }

    history["admissions"].append(admission_record)


# In[6]:


cxr_dataset = load_dataset("Yamini-1628/MIMIC-CXR-RRG", "findings_section", split="test")


# In[7]:


# Filter to this subject
cxr_list = cxr_dataset.filter(lambda r: r["subject_id"] == str(SUBJECT_ID))

# Prepare simple list of X-rays
cxrs = []
for r in cxr_list:
    image_filename = f"cxrs/{r['dicom_id']}.jpg"

    # Save PIL image to file
    r["main_image"].save(image_filename)

    cxrs.append({
        "study_id": r.get("study_id"),
        "dicom_id": r.get("dicom_id"),
        "findings": r.get("findings_section"),
        "impression": r.get("impression_section"),
        "main_image_path": image_filename,   # ← save path instead
        "study_date": r.get("StudyDate"),
        "study_time": r.get("StudyTime")
    })


# In[8]:


history["xray_studies"] = cxrs


# In[9]:


with open(f"patient_{SUBJECT_ID}_history.json", "w") as f:
    json.dump(history, f, indent=2)


# In[ ]:





# In[ ]:




