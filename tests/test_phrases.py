from src.kg.phrases import candidate_phrases, top_salient, entity_index, stem


def test_phrases_do_not_cross_stopwords_or_punctuation():
    s = candidate_phrases("Kernel ridge regression of the polynomial features. Speech recognition, robust training")
    assert "kernel ridge" in s and "ridge regression" in s and "kernel ridge regression" in s
    assert "polynomial feature" in s          # stem số nhiều
    assert not any("regression of" in p or "feature speech" in p for p in s)   # không vượt stopword/dấu câu
    assert "speech recognition" in s


def test_stem_and_digits():
    assert stem("models") == "model" and stem("class") == "class" and stem("bus") == "bus"
    assert not any(p.split()[0].isdigit() for p in candidate_phrases("2024 gaussian process 12 regression"))


def test_entity_index_df_window_and_top_salient():
    data = {"a": {"x y", "common z"}, "b": {"x y", "common z"}, "c": {"common z"}, "d": {"only q"}}
    ent = entity_index(data, min_df=2, max_df=2)
    assert ent["a"] == {"x y"} and ent["c"] == set() and ent["d"] == set()   # df=3 quá phổ biến, df=1 không nối được
    kept = top_salient(data, 1)
    assert kept["a"] == {"x y"}  # idf cao hơn 'common z'
