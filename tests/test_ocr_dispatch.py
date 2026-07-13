from OcrService import OcrService

def test_extract_text_dispatch_image(monkeypatch, tmp_path):
    ocr = OcrService()

    fake_img = tmp_path / "test.png"
    fake_img.write_text("dummy")

    monkeypatch.setattr(
        ocr,
        "_extract_from_image",
        lambda _: "IMAGE_TEXT"
    )

    text = ocr.extract_text(str(fake_img))
    assert text == "IMAGE_TEXT"


def test_extract_text_dispatch_pdf(monkeypatch, tmp_path):
    ocr = OcrService()

    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_text("dummy")

    monkeypatch.setattr(
        ocr,
        "_extract_from_pdf",
        lambda _: "PDF_TEXT"
    )

    text = ocr.extract_text(str(fake_pdf))
    assert text == "PDF_TEXT"
