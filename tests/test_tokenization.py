import json

import pytest

from llmfs.tokenization.bpe import ByteBPETokenizer, _merge_pair
from llmfs.tokenization.char import CharTokenizer
from llmfs.tokenization.io import load_tokenizer, save_tokenizer


def test_character_tokenizer_roundtrip(tmp_path):
    t = CharTokenizer.train(["café", "abcé"])
    assert t.decode(t.encode("café")) == "café"
    assert t.decode(t.encode("ca") + [t.eos_id] + t.encode("fé")) == "café"
    assert t.decode([t.eos_id], skip_special=False) == "<eos>"
    with pytest.raises(ValueError, match="absent"):
        t.encode("☃")
    save_tokenizer(t, tmp_path / "char.json")
    assert load_tokenizer(tmp_path / "char.json").encode("café") == t.encode("café")


def test_bpe_utf8_and_serialization(tmp_path):
    texts = ["hello 🐉🐉!", "été and 東京", "hello hello hello", "", "abcabc"]
    t = ByteBPETokenizer.train(texts, vocab_size=280)
    for word in texts + ["unseen 🎨 test", "", "\n \t"]:
        assert t.decode(t.encode(word)) == word
    assert t.encode("hello") == ByteBPETokenizer.train(texts, vocab_size=280).encode("hello")
    assert t.vocab_size <= 280
    save_tokenizer(t, tmp_path / "tok.json")
    loaded = load_tokenizer(tmp_path / "tok.json")
    assert loaded.encode("été 🐉!") == t.encode("été 🐉!")
    data = json.loads((tmp_path / "tok.json").read_text())
    data["vocab_base64"][-1] = "aA=="
    (tmp_path / "broken.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disagree"):
        load_tokenizer(tmp_path / "broken.json")


def test_bpe_known_first_merge_tie_break_and_document_boundary():
    t = ByteBPETokenizer.train(["abab", "abab"], vocab_size=258, min_pair_count=1)
    assert t.merges == [(ord("a"), ord("b"))]
    assert t.encode("abab") == [257, 257]
    assert _merge_pair([1, 1, 1], (1, 1), 5) == [5, 1]
    # Pairs are not counted across separate source documents.
    t2 = ByteBPETokenizer.train(["x", "y"], vocab_size=270, min_pair_count=1)
    assert t2.merges == []


def test_invalid_bpe_input():
    with pytest.raises(ValueError):
        ByteBPETokenizer.train([], vocab_size=300)
    with pytest.raises(ValueError):
        ByteBPETokenizer.train(["hey"], vocab_size=256)
    with pytest.raises(ValueError, match="Invalid"):
        ByteBPETokenizer().decode([10000])


def test_heap_bpe_matches_reference_over_random_unicode_and_repeated_pairs():
    import random
    from llmfs.tokenization.bpe import ByteBPETokenizer
    rng = random.Random(33)
    train = ["abababab ☀☀ zigzag ©", "ñ😂🎉 café abc abc abc", "bbbbb aaaaa bbbba"]
    tok = ByteBPETokenizer.train(train, vocab_size=290, min_pair_count=1)
    for i in range(150):
        raw = ''.join(rng.choice('abcñ☀😂  123') for _ in range(rng.randrange(65)))
        assert tok.encode(raw) == tok.encode_reference(raw), (i, raw)
    for raw in train + ["", "ababababab", "ééééééé"]:
        assert tok.encode(raw) == tok.encode_reference(raw)
