import os

import torch
from transformers import pipeline

LLM_MODEL = os.environ.get("CLINICIAN_LLM", "google/gemma-4-26B-A4B-it")
ASR_MODEL = os.environ.get("CLINICIAN_ASR", "google/medasr")
# Default 2-GPU layout: Gemma (heavy) alone on cuda:0; the small models
# (ASR + both embedders) share cuda:1. Override with env vars on boxes
# with more or fewer GPUs.
LLM_DEVICE = os.environ.get("CLINICIAN_LLM_DEVICE", "cuda:0")
ASR_DEVICE = os.environ.get("CLINICIAN_ASR_DEVICE", "cuda:1")

PIPE = pipeline(
    "image-text-to-text",
    model=LLM_MODEL,
    dtype=torch.bfloat16,
    device=LLM_DEVICE,
    max_new_tokens=8000,
)

PIPE_ASR = pipeline("automatic-speech-recognition", ASR_MODEL, device=ASR_DEVICE)
