from src.data import load_data
from src.preprocess import clean_text


def test_placeholders_replace_numbers_and_links():
    out = clean_text("Call 09061701461 now to claim £1000! Visit www.prize.com")
    assert "phonetoken" in out
    assert "moneytoken" in out
    assert "urltoken" in out
    assert "09061701461" not in out


def test_stopwords_removed_and_words_stemmed():
    out = clean_text("You are the winner of the winning prizes")
    assert "the" not in out.split()
    assert "prize" in out.split()  # "prizes" -> "prize"
    assert "win" in out.split()  # "winning" -> "win"


def test_dataset_has_no_duplicate_texts():
    df = load_data()
    assert df["text"].duplicated().sum() == 0
    assert 0.1 < df["is_spam"].mean() < 0.2


def test_model_on_obvious_examples():
    from src.predict import classify

    spam, ham = classify([
        "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot! Txt CLAIM to 81010",
        "Ok, I'll pick you up from the station at 7",
    ])
    assert spam[0] == "SPAM"
    assert ham[0] == "ham"
