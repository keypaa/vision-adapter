import pathlib


def test_no_stale_modal_pipeline_refs():
    for p in pathlib.Path("docs").rglob("*.md"):
        if "PIPELINE" in p.name:
            continue  # new doc may reference historically
        text = p.read_text()
        assert "modal_pipeline.py" not in text, f"{p} still references deleted file"
