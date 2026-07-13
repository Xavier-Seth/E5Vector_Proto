import pytest
from OcrService import OcrService


def test_extract_image_failure(monkeypatch):
    ocr = OcrService()

    def crash(_):
        raise RuntimeError("OCR failed")

    monkeypatch.setattr(ocr, "_extract_from_image", crash)

    text = ocr.extract_text("bad.png")
    assert text == ""
