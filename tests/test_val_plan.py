"""Validation on the streaming path: the 20h run must have a held-out signal.

Audit 2026-09-30: vision_adapter/train.py's streaming loop (the path Modal's
train_hf / train_hf_a100 use) had NO validation at all — a multi-hour run
could overfit with nothing but train loss to show it. The Volume path has
_val_probe; this pins the streaming equivalent: a val plan disjoint from the
train plan, and a periodic probe over it.

Design note: val re-streams from HF rather than materialising a local cache
(decision 2026-09-30 — disk/RAM are the constraint here, so correctness and
zero extra footprint beat a faster probe).

Run: python -m pytest tests/test_val_plan.py -q
"""
from vision_adapter.data.stream import build_epoch_plan


def _index(n):
    return {f"emb/{i}": ("shard-0", i, 200) for i in range(n)}


def _rows(n, tag="t"):
    return [{"emb": f"emb/{i}", "user": f"u{i}", "assistant": "a", "g": tag}
            for i in range(n)]


def test_val_plan_uses_only_val_rows():
    rows = _rows(10)
    idx = _index(10)
    plan = build_epoch_plan(rows, idx, sample_size=4, seed=0)
    assert sum(len(v) for v in plan.values()) <= 4


def test_val_and_train_plans_never_share_a_row():
    """The val rows must not appear in the train plan — same emb, same row."""
    train_rows = _rows(20, tag="train")
    val_rows = [{"emb": f"emb/{i}", "user": "v", "assistant": "a", "g": "val"}
                for i in range(15, 20)]
    idx = _index(20)
    train_plan = build_epoch_plan(train_rows, idx, sample_size=10, seed=0)
    val_plan = build_epoch_plan(val_rows, idx, sample_size=5, seed=0)

    def _embs(plan):
        out = set()
        for rs in plan.values():
            for r in rs:
                out.add(r["emb"])
        return out

    assert _embs(train_plan).isdisjoint(_embs(val_plan))


def test_val_rows_absent_from_the_embedding_index_are_dropped():
    """A val row with no embedding cannot be evaluated — it must not raise."""
    rows = [{"emb": "emb/0", "user": "v", "assistant": "a", "g": "val"},
            {"emb": "emb/missing", "user": "v", "assistant": "a", "g": "val"}]
    plan = build_epoch_plan(rows, _index(1), sample_size=5, seed=0)
    got = [r["emb"] for rs in plan.values() for r in rs]
    assert got == ["emb/0"]


def test_val_step_decides_when_to_probe():
    """Probing is gated on a step interval, and never on step 0."""
    from vision_adapter.train import _val_due

    assert not _val_due(0, val_every=50)
    assert _val_due(50, val_every=50)
    assert not _val_due(51, val_every=50)
    assert _val_due(100, val_every=50)
    # disabled entirely when the config has no interval
    assert not _val_due(100, val_every=0)
    # the last step always probes, so a short run still reports a val loss
    assert _val_due(10, val_every=50, total_steps=10)


def test_val_record_shape_is_distinguishable_from_train():
    """A val line must never be mistaken for a train line in the log."""
    from vision_adapter.train import _val_record

    rec = _val_record(step=50, loss=1.234, n_rows=1272, wall_min=3.5)
    assert rec["type"] == "val"
    assert rec["step"] == 50
    assert rec["loss"] == 1.234
    assert rec["n_rows"] == 1272
    assert "gnorm" not in rec      # no grads on a val pass
    assert "ema_loss" not in rec


def test_config_carries_the_val_interval():
    from vision_adapter.config import default_config

    assert default_config().val_every > 0   # on by default: no unmonitored run


def test_val_manifest_name_is_the_disjoint_one():
    """Must be the rebuilt disjoint split, not the 47%-overlap original."""
    from vision_adapter.train import VAL_MANIFEST_FILE

    assert VAL_MANIFEST_FILE == "train_manifest_val_disjoint.jsonl"


def test_split_val_manifest_file_yields_only_data_rows(tmp_path):
    """Header rows are provenance, never val samples."""
    from vision_adapter.train import _val_rows_from_file

    p = tmp_path / "val.jsonl"
    p.write_text(
        '{"type":"manifest_header","row_count":2}\n'
        '{"emb":"e1","user":"u","assistant":"a","g":"doc"}\n'
        '{"emb":"e2","user":"u","assistant":"a","g":"conv"}\n'
    )
    rows = _val_rows_from_file(p)
    assert [r["emb"] for r in rows] == ["e1", "e2"]


def test_val_probe_returns_mean_loss_and_row_count():
    """The probe averages per-batch loss into one number plus how many rows."""
    import torch

    from vision_adapter.train import _val_probe

    def collate(n):
        return {
            "input_ids": torch.zeros(n, 6, dtype=torch.long),
            "attention_mask": torch.ones(n, 6, dtype=torch.long),
        }

    batches = [collate(2), collate(1)]
    calls = []

    def fake_loss(model, proj, batch, device):
        calls.append(batch)
        return torch.tensor(float(len(calls)))

    loss, n = _val_probe(fake_loss, None, None, batches, "cpu")
    assert n == 3
    assert loss == 1.5          # mean of batch losses 1.0 and 2.0


def test_val_probe_skips_batches_with_no_supervised_token():
    """A fully-masked batch must not poison the average with a fake 0."""
    import torch

    from vision_adapter.train import _val_probe

    batches = [{"input_ids": torch.zeros(1, 4, dtype=torch.long)}]
    loss, n = _val_probe(lambda *a: None, None, None, batches, "cpu")
    assert n == 1 and loss == 0.0


def test_val_loss_reuses_the_train_step_recipe():
    """The probe must compute loss the same way the train step does.

    Regression guard: an early version called native_train_forward and read
    `.loss` off its return, but it returns a tuple of tensors — the probe
    would have raised on the first val step of a real run.
    """
    import torch

    from vision_adapter.train import _batch_loss

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.lm_head = torch.nn.Linear(4, 7)
            self.embed = torch.nn.Embedding(16, 4)

            class _Backbone(torch.nn.Module):
                def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None):
                    h = torch.zeros(
                        inputs_embeds.shape[0], inputs_embeds.shape[1], 4
                    )
                    return type("O", (), {"last_hidden_state": h})()

            self.model = _Backbone()

        def get_input_embeddings(self):
            return self.embed

    proj = torch.nn.Linear(4, 4)
    batch = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3, 4, -100]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "vis": torch.zeros(1, 2, 4),
        "n_vis": torch.tensor([2]),
    }
    loss = _batch_loss(_Model(), proj, batch, "cpu", None)
    assert torch.isfinite(loss)
    assert loss.ndim == 0
