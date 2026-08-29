import torch

from vision_adapter.backends.base import get_backend
from vision_adapter.backends.local import LocalBackend


def test_local_backend_roundtrip(tmp_path):
    b = LocalBackend(tmp_path)
    t = torch.randn(3, 4096, dtype=torch.bfloat16)
    b.write_embedding("embeddings/abc123.pt", t)
    assert b.exists("embeddings/abc123.pt")
    rt = b.read_embedding("embeddings/abc123.pt")
    assert torch.equal(rt, t)


def test_get_backend_factory(tmp_path):
    b = get_backend("local", root=tmp_path)
    assert isinstance(b, LocalBackend)


def test_modal_backend_helpful_error():
    from vision_adapter.backends.modal import ModalBackend

    # Import must not crash — only instantiation when modal missing should error helpfully.
    # If modal is installed, instantiation should succeed (or at least not raise RuntimeError about install).
    try:
        import modal  # noqa: F401

        # modal is installed — just verify construction doesn't raise the "not installed" error
        try:
            mb = ModalBackend(volume_name="vision-adapter-test-does-not-matter")
            assert mb is not None
        except Exception as e:
            # Any error other than the "pip install modal" message is acceptable (e.g. auth)
            assert "pip install modal" not in str(e)
    except ImportError:
        # modal not installed — must raise with helpful message
        try:
            ModalBackend()
            assert False, "expected RuntimeError when modal missing"
        except RuntimeError as e:
            assert "pip install modal" in str(e) or "modal" in str(e).lower()
