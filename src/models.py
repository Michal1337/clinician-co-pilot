import torch
from transformers import pipeline

PIPE = pipeline(
    "image-text-to-text",
    model="google/gemma-4-31B-it",
    dtype=torch.bfloat16,
    device="cuda:2",
    max_new_tokens=32000,
    max_length=None,
)

PIPE_ASR = pipeline("automatic-speech-recognition", "google/medasr", device="cuda:3")
