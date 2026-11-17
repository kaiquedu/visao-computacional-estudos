from __future__ import annotations

import os
from typing import Protocol

from PIL import Image

CAPTION_MODEL = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-tc-big-en-pt")


class Captioner(Protocol):
    def describe(self, image: Image.Image) -> str: ...


class BlipCaptioner:
    def __init__(self, caption_model: str = CAPTION_MODEL, translation_model: str = TRANSLATION_MODEL):
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor, MarianMTModel, MarianTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(caption_model)
        self.model = BlipForConditionalGeneration.from_pretrained(caption_model).to(self.device).eval()
        self.tokenizer = MarianTokenizer.from_pretrained(translation_model)
        self.translator = MarianMTModel.from_pretrained(translation_model).to(self.device).eval()

    def caption(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=50, num_beams=3, early_stopping=True)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def translate(self, text: str) -> str:
        inputs = self.tokenizer([f">>pob<< {text}"], return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.translator.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def describe(self, image: Image.Image) -> str:
        return self.translate(self.caption(image))
