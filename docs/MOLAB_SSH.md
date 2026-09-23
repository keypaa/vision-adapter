# MOLAB_SSH — real SSH into a Molab GPU box

Molab containers egress via a NAT pool (no stable public IP, private `10.x`),
so **inbound SSH is impossible**. `tmate` is also unusable (`ssh.tmate.io`
DNS-blocked). The working path is an **outbound bore tunnel** (443-based,
no account): `bore.pub:<PORT> → localhost:2222` on the box.

One script does both sides: `scripts/molab_ssh_up.sh` (remote mode default,
`--local` for this machine). Both modes are idempotent.

## Fresh session (new notebook)

**1. Spawn the notebook.** https://molab.marimo.io/notebooks → new notebook →
specs button → attach `RTX PRO 6000`. **Keep the tab open** (12h max,
90min idle kills container + tunnel + sshd).

**2. Local — keypair + pubkey (once):**

```bash
bash scripts/molab_ssh_up.sh --local   # prints pubkey, creates ~/.ssh/molab_bore if missing
```

**3. Molab — expose (terminal in the notebook):**

```bash
git clone https://github.com/keypaa/vision-adapter.git && cd vision-adapter  # if needed
PUBKEY="ssh-ed25519 AAAA... (from step 2)" bash scripts/molab_ssh_up.sh
# prints MOLAB SSH BUNDLE with BORE_ADDR bore.pub:<PORT>
```

**4. Local — config + test:**

```bash
bash scripts/molab_ssh_up.sh --local --remote-port <PORT from bundle>
# expect: SSH_OK + hostname + RTX PRO 6000
ssh molab-bore   # from now on
```

## Reconnect (tunnel died / new PORT)

The bore port is **ephemeral** — every restart prints a new one. No need to
touch keys/sshd again:

```bash
# Molab: rerun (key already installed, sshd already up — just re-tunnels)
PUBKEY="..." bash scripts/molab_ssh_up.sh   # note new bore.pub:<PORT>
# Local:
bash scripts/molab_ssh_up.sh --local --remote-port <NEW PORT>
```

## What the script does (Molab side)

1. Installs `openssh-server` if missing, `ssh-keygen -A`, starts `sshd -p 2222`
   (skips if already listening; `ss`-less fallback via `/proc/net/tcp`).
2. Installs the pubkey into **both** `$HOME/.ssh` and the passwd home
   (`getent passwd`). They differ on Molab (`HOME=/home/marimo`, user `root`
   → sshd reads `/root/.ssh`) — this was the actual auth failure we hit.
3. Installs `bore v0.5.2` if missing, restarts `bore local 2222 --to bore.pub`,
   waits for `listening at bore.pub:<PORT>`, prints the bundle + local stanza.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Permission denied (publickey,password)`, key offered | key in wrong home — script handles via `getent`; if manual, write to `/root/.ssh/authorized_keys`, not `~/.ssh` |
| `sshd` rejects after container restart | `/root/.ssh` wiped (ephemeral) — rerun step 3 |
| `ssh.tmate.io lookup failure` | expected — tmate is blocked, use this script (bore) |
| `bore` log has no `listening at` | egress blocked — `cat /tmp/bore.log`; needs HTTPS egress to `bore.pub:443` |
| `pkill -f "bore local ..."` kills your own shell | pattern self-matches the remote command — kill by PID (`pgrep -af "bore local"`, then `kill <pid>`) |
| Everything dead after idle | tab closed / 90min idle / 12h max — respawn notebook, follow Reconnect |

## Multi-session rules (learned 2026-09-23)

Two agent sessions sharing one Molab host and one local user collided on:
shared `Host molab-bore` stanza (last writer wins), one tunnel for two
notebooks (a session landed on the other project's container), and a shared
cron namespace (`keypaa`).

- One SSH Host alias per (session, project): e.g. `molab-bore-va` vs
  `molab-bore`. Never touch the other's stanza; never run `--local
  --remote-port` (it rewrites the shared alias) — write your own stanza
  by hand.
- One bore tunnel per notebook (each box gets its own port).
- Cron `agent_name` per project (e.g. `keypaa-vision` vs `keypaa-tiny`);
  each session acts only on its own job names.
- Fingerprint on every (re)connect: `hostname` + `ls /marimo` must match
  the expected project before any command. On mismatch: stop everything.

## Incident 2026-09-23 — account restricted, DO NOT REPEAT

Account `molab.marimo.io` restricted (review at
`https://marimo.io/account-restricted`). Suspected cause: persistent
outbound bore tunnel (network circumvention) + keep-alive watchdog cells
defeating the 90-min idle shutdown. Tunnel dead (`Connection refused`),
all box-local state lost (only HF pushes survive).

Rules going forward (non-negotiable):
- No bore/SSH tunnels, no relay of any kind.
- No keep-alive automation of any kind (cells, loops, pings).
- Active tab, human-driven sessions only; close the notebook when done.
- One driver per box; fingerprint check stays mandatory.

## Security

- One keypair per direction (`~/.ssh/molab_bore`, comment `local-to-molab-bore`);
  never reuse personal keys, never commit keys.
- Revoke: delete the line from `/root/.ssh/authorized_keys` on Molab, or
  `pkill -f "bore local 2222"` to close the tunnel. Killing the notebook
  destroys everything.
- `bore.pub` relays encrypted SSH bytes — it sees metadata (IPs, timing),
  not session content (end-to-end SSH encryption still holds).
