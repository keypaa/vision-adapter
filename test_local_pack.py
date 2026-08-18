# test_local_pack.py
from local_pack import sorted_embedding_names, shard_slices, FileEntryLike


def test_sorted_embedding_names_is_sorted_and_strips_prefix():
    entries = [
        FileEntryLike(path="embeddings/zzz.pt"),
        FileEntryLike(path="embeddings/aaa.pt"),
        FileEntryLike(path="embeddings/mmm.pt"),
    ]
    assert sorted_embedding_names(entries) == [
        "embeddings/aaa.pt",
        "embeddings/mmm.pt",
        "embeddings/zzz.pt",
    ]


def test_shard_slices_boundaries():
    names = [f"embeddings/{i:04d}.pt" for i in range(7)]
    s = shard_slices(names, 3)
    assert s == [
        ["embeddings/0000.pt", "embeddings/0001.pt", "embeddings/0002.pt"],
        ["embeddings/0003.pt", "embeddings/0004.pt", "embeddings/0005.pt"],
        ["embeddings/0006.pt"],
    ]