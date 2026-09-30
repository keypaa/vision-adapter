"""Precompute wiring: CLI run_precompute must actually generate embeddings.

History: the staged CLI wired to a validation-only stub while the real MoonViT
inference lived in precompute_colab (docs even claimed the reverse). This pins
end-to-end wiring on a local backend with one tiny image and a stubbed ViT
(no weights download, no GPU). Run:
    python -m pytest tests/test_precompute_wiring.py -q
"""

import torch
from PIL import Image

import pytest

from vision_adapter.backends.local import LocalBackend
from vision_adapter.models import precompute_colab as pc
from vision_adapter.models.precompute import _emb_key, run_precompute


class _FakeVit:
    def __init__(self):
        from types import SimpleNamespace

        self.patch_embed = SimpleNamespace(
            proj=SimpleNamespace(weight=torch.empty(0, dtype=torch.float32))
        )

    def __call__(self, pixel_values, grid_thws):
        return torch.randn(pixel_values.shape[0], 4, 4096)


def test_run_precompute_writes_embedding(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "load_vit", lambda *a, **k: _FakeVit())
    img_dir = tmp_path / "images" / "agentic"
    img_dir.mkdir(parents=True)
    Image.new("RGB", (56, 56), "white").save(img_dir / "t.png")
    run_precompute(LocalBackend(tmp_path), tmp_path, patch_cap=262144, device="cpu")
    out = tmp_path / "embeddings" / _emb_key("agentic/t.png")
    assert out.is_file(), "run_precompute must write the embedding file"
    ten = torch.load(str(out), map_location="cpu", weights_only=True)
    assert ten.dim() == 2 and ten.shape[-1] == 4096


def test_run_precompute_rejects_non_local_backend(tmp_path):
    (tmp_path / "images").mkdir(parents=True)

    class _Remote:
        pass

    with pytest.raises(ValueError, match="local"):
        run_precompute(_Remote(), tmp_path, patch_cap=1, device="cpu")
