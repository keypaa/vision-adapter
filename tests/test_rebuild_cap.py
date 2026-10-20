def test_the_budget_is_per_slice():
    """The reference's 300k-px rule was set for art photos; a scanned document
    needs more pixels or the answer is unreadable."""
    from scripts.rebuild_corpus import BUDGETS, budget_for, subset_of

    assert subset_of("cauldron/chartqa-0000001-0.png") == "cauldron/chartqa"
    assert subset_of("cauldron/docvqa-0001234-0.png") == "cauldron/docvqa"
    assert subset_of("agentic/waveui_013568.jpg") == "agentic/waveui"
    assert subset_of("agentic/miniwob_000585.png") == "agentic/miniwob"

    # the default tier is the reference's rule
    assert budget_for("agentic/waveui_013568.jpg", 300_000) == 300_000
    assert budget_for("cauldron/chartqa-0000001-0.png", 300_000) == 300_000
    # the document tier gets more pixels
    assert budget_for("cauldron/docvqa-0001234-0.png", 300_000) == 800_000
    assert BUDGETS["cauldron/websight"] == 800_000


def test_cap_image_floors_to_a_multiple_of_28():
    """The reference builder's rule: scale to the pixel budget, then FLOOR the
    dimensions to a multiple of 28 (our own navit_resize pads UP, which pushed
    the merged token count to 411-425 where the reference gets <= 382)."""
    from PIL import Image

    from scripts.rebuild_corpus import cap_image

    im = Image.new("RGB", (1000, 1000))
    out = cap_image(im, 300_000)
    w, h = out.size
    assert w % 28 == 0 and h % 28 == 0, (w, h)
    assert w * h <= 300_000, (w, h)
    assert (w // 28) * (h // 28) <= 300_000 // 196 + 1


def test_cap_image_leaves_small_images_alone():
    from PIL import Image

    from scripts.rebuild_corpus import cap_image

    im = Image.new("RGB", (280, 280))
    assert cap_image(im, 300_000).size == (280, 280)


def test_cap_image_scales_a_wide_image():
    from PIL import Image

    from scripts.rebuild_corpus import cap_image

    im = Image.new("RGB", (4000, 1000))
    out = cap_image(im, 300_000)
    w, h = out.size
    assert w * h <= 300_000, (w, h)
    assert w % 28 == 0 and h % 28 == 0
