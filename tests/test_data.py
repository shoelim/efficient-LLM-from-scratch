import numpy as np
import pytest
import torch

from llmfs.data.batches import get_batch
from llmfs.data.prepare import (dataset_fingerprint, load_tokens, prepare_data,
                                read_documents, split_documents)
from llmfs.tokenization.io import load_tokenizer


def test_document_split_and_dedup(tmp_path):
    source = tmp_path / "text.txt"
    source.write_text("alpha\n\nbeta\n\nalpha\n\ngamma\n\ndelta")
    docs = read_documents(source)
    assert len(docs) == 4
    train, val = split_documents(docs, 0.25, seed=3)
    assert set(train).isdisjoint(val)
    assert len(train) == 3 and len(val) == 1
    assert split_documents(docs, 0.25, seed=3) == (train, val)


def test_prepare_and_fingerprint(tmp_path):
    text = tmp_path / "documents.txt"
    text.write_text("abababab\n\nababcccc\n\nababdddd\n\nababeefg")
    out = tmp_path / "prepared"
    m = prepare_data(text, out, tokenizer_type="bpe", vocab_size=265)
    assert m["n_train_documents"] == 3
    assert m["n_val_documents"] == 1
    assert load_tokenizer(out / "tokenizer.json").vocab_size == m["vocab_size"]
    assert load_tokens(out, "train").dtype == np.uint32
    assert len(dataset_fingerprint(out)) == 64
    with (out / "tokenizer.json").open("a") as file:
        file.write(" ")
    with pytest.raises(ValueError, match="changed"):
        dataset_fingerprint(out)


def test_batch_shift_and_valid_indices():
    tokens = np.arange(30, dtype=np.uint32)
    generator = torch.Generator().manual_seed(14)
    x, y = get_batch(tokens, 8, 5, generator, torch.device("cpu"))
    assert x.shape == (5, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y[:, -1], x[:, -1] + 1)
    with pytest.raises(ValueError):
        get_batch(tokens[:8], 8, 5, generator, torch.device("cpu"))
