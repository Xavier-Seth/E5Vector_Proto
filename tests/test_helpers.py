from app import clean_text, word_count

def test_clean_text_basic():
    raw = "Hello   world\n\n\nTest"
    cleaned = clean_text(raw)
    assert cleaned == "Hello world\n\nTest"

def test_word_count_basic():
    text = "one two three"
    assert word_count(text) == 3
