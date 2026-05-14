import os
from typing import List

import numpy as np
import torch
from langchain.embeddings.base import Embeddings
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from model_paths import resolve_model

# Both embedders are small; co-locate them on the non-LLM GPU by default
# (cuda:1) so cuda:0 is free for Gemma. Override via env var if needed.
EMBED_DEVICE = os.environ.get("CLINICIAN_EMBED_DEVICE", "cuda:1")


class MedSigLIPEmbeddings(Embeddings):
    def __init__(self, model_name="google/medsiglip-448", device="cuda"):
        self.device = device
        resolved = resolve_model(model_name)
        self.processor = AutoProcessor.from_pretrained(resolved)
        self.model = AutoModel.from_pretrained(resolved).to(self.device)
        self.model.eval()

    def embed_documents(self, image_paths):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)["pooler_output"]

        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        return embeddings.cpu().numpy()

    def embed_query(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(
            self.device
        )

        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)["pooler_output"]

        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        return embeddings.cpu().numpy()[0]


class ClinicalBERTEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 8,
    ):
        resolved = resolve_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModel.from_pretrained(resolved).to(device)
        self.model.eval()

        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    def _mean_pooling(self, last_hidden_state, attention_mask):
        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def _embed_batch(self, texts: List[str]):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state

            embeddings = self._mean_pooling(last_hidden, inputs["attention_mask"])

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().float().numpy()

    def _embed(self, texts: List[str]):
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def embed_documents(self, texts: List[str]):
        return self._embed(texts)

    def embed_query(self, text: str):
        return self._embed([text])[0]


TEXT_EMBEDDINGS = ClinicalBERTEmbeddings(device=EMBED_DEVICE)
IMAGE_EMBEDDINGS = MedSigLIPEmbeddings(device=EMBED_DEVICE)
