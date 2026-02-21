import torch
from transformers import pipeline

PIPE = pipeline(
    "image-text-to-text",
    model="../../medgemma4b",
    dtype=torch.bfloat16,
    device="cuda:1",
    max_new_tokens=32000,
    max_length = None
)
PIPE_TXGEMMA = pipeline(
    "text-generation",
    model="../../medgemma4b",
    device="cpu",
    max_new_tokens=32000,
    max_length = None
)

PIPE_ASR  = pipeline("automatic-speech-recognition", "../../medasr", device="cuda:2")