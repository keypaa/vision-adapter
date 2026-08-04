import pytest
import torch
from PIL import Image

from preprocess import collate_images, process_image


def _solid(w, h, value):
    return Image.new("RGB", (w, h), (value, value, value))


def _expected_naive_values(scale):
    from PIL import Image as _I

    img = _I.new("RGB", (14, 14))
    for y in range(14):
        for x in range(14):
            img.putpixel((x, y), (x * 17, y * 17, (x + y) * 8))
    return img.resize((2, 2), Image.BICUBIC)


def test_1000x1000_stays_grid_72_72():
    out = process_image(_solid(1000, 1000, 128))
    grid = out["grid_thws"]
    assert grid.dtype == torch.long
    assert grid.shape == (1, 3)
    assert grid.tolist() == [[1, 72, 72]]  # 1000 -> pad 1008, 1008/14 = 72
    assert out["pixel_values"].shape == (5184, 3, 14, 14)
    assert out["pixel_values"].dtype == torch.float32


def test_50000x100_capped_to_max_side_7168():
    out = process_image(_solid(50000, 100, 64))
    t, h, w = out["grid_thws"][0].tolist()
    assert t == 1
    assert w * 14 <= 7168
    assert h * 14 <= 7168
    # resize was significant: (50000//14)*(100//14) = 25004 > 65536 patches
    # scale = sqrt(65536/25004) -> new_w ~ 7168-capped
    assert h >= 1 and w >= 1
    assert out["pixel_values"].shape[0] == h * w


def test_normalization_endpoints():
    black = process_image(_solid(28, 28, 0))["pixel_values"]
    white = process_image(_solid(28, 28, 255))["pixel_values"]
    assert torch.allclose(black, torch.full_like(black, -1.0), atol=2e-3)
    assert torch.allclose(white, torch.full_like(white, 1.0), atol=2e-3)


def test_collate_packs_two_images():
    out = collate_images([_solid(1000, 1000, 128), _solid(50000, 100, 64)])
    grid = out["grid_thws"]
    assert grid.shape == (2, 3)
    s1 = process_image(_solid(1000, 1000, 128))
    s2 = process_image(_solid(50000, 100, 64))
    assert grid.tolist() == [s1["grid_thws"][0].tolist(), s2["grid_thws"][0].tolist()]
    n_expected = int(grid[0, 1] * grid[0, 2] + grid[1, 1] * grid[1, 2])
    assert out["pixel_values"].shape[0] == n_expected
    # packed pixel_values equals concatenation of singles
    assert torch.equal(
        out["pixel_values"],
        torch.cat([s1["pixel_values"], s2["pixel_values"]], dim=0),
    )


def test_deterministic_no_io(tmp_path, monkeypatch):
    # no file I/O beyond reading the input PIL image: forbid open()
    import builtins

    real_open = builtins.open

    def guarded(*a, **k):
        raise AssertionError("preprocess must not perform file I/O")

    img = _solid(200, 140, 30)
    a = process_image(img)
    monkeypatch.setattr(builtins, "open", guarded)
    b = process_image(img)  # must not touch open() for files
    assert torch.equal(a["pixel_values"], b["pixel_values"])
    assert torch.equal(a["grid_thws"], b["grid_thws"])
