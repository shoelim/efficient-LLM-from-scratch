import json
from pathlib import Path

import pytest
import torch

from llmfs.cli import load_inference_model
from llmfs.data.prepare import prepare_data
from llmfs.inference.export import export_inference, load_export
from llmfs.training.train import run_training


@pytest.mark.parametrize("fused", [True, False])
def test_safetensors_inference_export_exact_weights(tmp_path, fused):
    import yaml
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / 'data'
    prepare_data(root / 'examples/toy_corpus.txt', data, tokenizer_type='char')
    conf = yaml.safe_load((root / 'configs/smoke.yaml').read_text())
    conf['model']['fused_qkv'] = fused
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(conf))
    checkpoint = run_training(data, path, tmp_path / 'run')
    directory = export_inference(checkpoint, tmp_path / 'export')
    assert (directory / 'model.safetensors').is_file()
    x = torch.randint(0, 17, (1, 5))
    a = load_inference_model(checkpoint, torch.device('cpu'))
    b = load_export(directory, torch.device('cpu'))
    x = x % a.config.vocab_size
    assert torch.equal(a(x).logits, b(x).logits)
    assert a.token_embedding.weight.data_ptr() == a.lm_head.weight.data_ptr()
    assert b.token_embedding.weight.data_ptr() == b.lm_head.weight.data_ptr()
    config = json.loads((directory / 'config.json').read_text())
    config['format_version'] = 999
    (directory / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_export(directory, torch.device('cpu'))
