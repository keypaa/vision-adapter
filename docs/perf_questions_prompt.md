# Data-loading performance — fresh session, symptoms only

Copy everything below into a new session with no prior context.

---

You are looking at a training run that is **far too slow**, and the data
loading path is the suspect. There is a codebase to read and a proposal to
make. Do not start implementing — investigate, then recommend.

## The project

A vision→LLM adapter trainer. It learns a small projector that maps frozen
image embeddings (already precomputed, ~1 TB of them) into a language
backbone. Only the projector is trained; the backbone is frozen. Training
runs on rented single GPUs.

Training data lives in a Hugging Face dataset repo as parquet shards:

- ~103 shards, 136,000 rows total, ~830 GB of parquet
- each row is one image embedding: `key`, `n_vis` (visual token count, 16
  to 16,653, median 364), `vis_bytes` (`n_vis × 4096` bfloat16)
- the rows a run actually needs are listed in a JSONL manifest, keyed by the
  same `key`

## What we observe

**1. Runs are slow, and unevenly so.** On a 4,000-step run the total wall
clock was ~6.8 hours. Steps are not uniform: most take ~6 seconds, but
clusters of steps take minutes each. Those slow clusters land
reproducibly on particular shards — the same step numbers are slow across
separate runs.

**2. The big shards are disproportionately slow.** The shard files are not
uniformly sized — they span from ~435 MB to ~64 GB, and the largest ~27 of
them hold about 83% of all the data. Slow steps correlate with the large
shards.

**3. Throughput on the network looks poor.** On one host the effective
transfer rate was ~10 MiB/s, far below the ~1 GiB/s the transfer library is
documented to reach. On another host, Range-based fetching reaches ~28 MiB/s
in aggregate across parallel connections.

**4. Memory blows up on a small machine.** A 2-step smoke test on a 15 GB
Colab box (12 GB usable) was killed by the OOM killer after ~30 minutes,
while working with a single 1.8 GB row group. Nothing about the model's own
footprint explains it. The same code runs fine on a large box.

**5. A trivial run downloads a lot.** A 2-step smoke test with a plan of 8
rows fetched well over 1 GB before it was killed.

## What we want

- Training that spends its time on the GPU, not waiting for bytes.
- No download of data we are not going to train on.
- A path that works on a 15 GB machine as well as on a large one, without a
  code branch per machine.
- Something that scales to the next order of magnitude of data without a
  rewrite.

## Constraints we care about

- The set of rows served, and their order, must stay stable across runs.
  Checkpoints record a hash of the plan and resume from a row offset, so a
  change in which rows are served or in what order invalidates resumes.
- The target platform is a rented GPU instance with plenty of disk, not a
  laptop. Assume ~500 GB of local disk and 1-10 Gb/s of network.
- We also want this to keep working when someone runs it on a free Colab
  T4 to check something quickly.

## Where to look

`vision_adapter/data/stream.py` — the streaming loader.
`vision_adapter/train.py` — where the dataset and its plan are wired up.

## What to hand back

1. **Where the time actually goes.** Quantify it: how much of a step is
   bytes arriving, bytes being copied locally, and bytes being decoded.
   If a run already has per-step timings, use them rather than reasoning
   about it.
2. **What is downloaded that is not used.** Quantify it on a plan of
   realistic size, and say whether it grows worse for small runs or large.
3. **The memory ceiling**, explained: what is resident at any moment for a
   given row-group size, and which of those residents are necessary.
4. **A recommended design.** Concrete and specific to this workload — where
   the bytes live, in what order, how they reach the consumer, what is
   cached and for how long. If there is a standard approach that fits, say
   so and say why it fits.
5. **What to do first**, cheapest change with the largest effect, and what
   it costs in complexity and in risk to the resume path.

Say plainly what you could not determine without a network, a GPU, or the
corpus. If a number is an estimate rather than a measurement, label it. If
you think one of the observations above has a simpler explanation than it
looks, say that too — we may be measuring the wrong thing.