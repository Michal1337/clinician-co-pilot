import os

import torch
from transformers import pipeline

LLM_MODEL = os.environ.get("CLINICIAN_LLM", "google/gemma-4-26B-A4B-it")
ASR_MODEL = os.environ.get("CLINICIAN_ASR", "google/medasr")
LLM_DEVICE = os.environ.get("CLINICIAN_LLM_DEVICE", "cuda:2")
ASR_DEVICE = os.environ.get("CLINICIAN_ASR_DEVICE", "cuda:3")

PIPE = pipeline(
    "image-text-to-text",
    model=LLM_MODEL,
    dtype=torch.bfloat16,
    device=LLM_DEVICE,
    max_new_tokens=8000,
)

PIPE_ASR = pipeline("automatic-speech-recognition", ASR_MODEL, device=ASR_DEVICE)
