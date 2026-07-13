from OcrService import OcrService

def test_clean_text_basic():
    ocr = OcrService()

    raw = "Hello   world\n\n\nTest"
    cleaned = ocr.clean_text(raw)

    assert cleaned == "Hello world\nTest"



def test_split_by_words():  
    ocr = OcrService()

    text = "one two three four five six seven eight nine ten"
    chunks = ocr._split_by_words(text, max_words=4)

    assert chunks == [
        "one two three four",
        "five six seven eight",
        "nine ten"
    ]


def test_split_to_snippets_min_words():
    ocr = OcrService()

    text = "one two three four five six seven eight nine ten"
    snippets = ocr.split_to_snippets(text, max_words=5)

    # Minimum snippet size is 15 words → should be empty
    assert snippets == []
