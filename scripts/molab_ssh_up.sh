#!/usr/bin/env bash
# Expose THIS Molab container via real SSH through a bore tunnel.
#
# Why this exists: Molab egresses via a NAT pool (no stable public IP,
# private 10.x), and tmate's ssh.tmate.io is DNS-blocked. bore tunnels
# over HTTPS-egress (port 443) and works. HOME may also differ from the
# passwd home (e.g. HOME=/home/marimo while USER=root), so the key is
# installed in BOTH places.
#
# Run ON MOLAB (as root) — exposes this box:
#   PUBKEY="ssh-ed25519 AAAA... local-to-molab" bash scripts/molab_ssh_up.sh
#   bash scripts/molab_ssh_up.sh --pubkey "ssh-ed25519 AAAA..." [--port 2222]
#
# Run ON THIS MACHINE — keypair + config + test:
#   bash scripts/molab_ssh_up.sh --local                       # step 1: prints pubkey for Molab
#   bash scripts/molab_ssh_up.sh --local --remote-port 61079   # step 3: config + SSH_OK test
#
# Full flow: docs/MOLAB_SSH.md.
# Keep the Molab tab open (12h max / 90min idle kills container + tunnel).
set -euo pipefail

PORT=2222
PUBKEY="${MOLAB_SSH_PUBKEY:-${PUBKEY:-}}"
BORE_VER="0.5.2"
BORE_BIN="./bore"
LOCAL_MODE=0
REMOTE_PORT_ARG=""
LOCAL_USER="root"
LOCAL_HOST="bore.pub"
LOCAL_KEY="$HOME/.ssh/molab_bore"

usage() {
  echo "Molab side: PUBKEY=\"ssh-ed25519 ...\" bash scripts/molab_ssh_up.sh [--pubkey KEY] [--port PORT]" >&2
  echo "Local side: bash scripts/molab_ssh_up.sh --local [--remote-port PORT] [--user USER] [--key PATH]" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pubkey) PUBKEY="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --local) LOCAL_MODE=1; shift ;;
    --remote-port) REMOTE_PORT_ARG="$2"; shift 2 ;;
    --user) LOCAL_USER="$2"; shift 2 ;;
    --host) LOCAL_HOST="$2"; shift 2 ;;
    --key) LOCAL_KEY="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

# --- LOCAL MODE: keypair + ~/.ssh/config + connectivity test --------------
if [[ "$LOCAL_MODE" -eq 1 ]]; then
  if [[ ! -f "$LOCAL_KEY" ]]; then
    echo "[local] generating $LOCAL_KEY..."
    ssh-keygen -t ed25519 -f "$LOCAL_KEY" -N "" -C "local-to-molab-bore"
  fi
  echo "[local] pubkey (paste to Molab side):"
  cat "$LOCAL_KEY.pub"
  [[ -n "$REMOTE_PORT_ARG" ]] || {
    echo "[local] next: on Molab run PUBKEY=\"\$(cat $LOCAL_KEY.pub)\" bash scripts/molab_ssh_up.sh"
    echo "[local] then rerun: bash scripts/molab_ssh_up.sh --local --remote-port <PORT from Molab bundle>"
    exit 0
  }
  # upsert Host molab-bore stanza: drop old block, append fresh (idempotent)
  touch "$HOME/.ssh/config"
  chmod 600 "$HOME/.ssh/config"
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$HOME/.ssh/config" <<'PYEOF'
import re, sys
p = sys.argv[1]
t = open(p).read()
t = re.sub(r"(?m)^Host molab-bore$\n(?:^[ \t]+.*\n?)*", "", t).rstrip("\n") + "\n"
open(p, "w").write(t)
PYEOF
  else
    awk 'BEGIN{skip=0} /^Host molab-bore$/{skip=1; next} /^Host /{skip=0} !skip{print}' \
      "$HOME/.ssh/config" > "$HOME/.ssh/config.tmp" && mv "$HOME/.ssh/config.tmp" "$HOME/.ssh/config"
  fi
  cat >> "$HOME/.ssh/config" <<EOF

Host molab-bore
  HostName $LOCAL_HOST
  Port $REMOTE_PORT_ARG
  User $LOCAL_USER
  IdentityFile $LOCAL_KEY
  StrictHostKeyChecking accept-new
EOF
  chmod 600 "$HOME/.ssh/config"
  echo "[local] config updated: $LOCAL_USER@$LOCAL_HOST:$REMOTE_PORT_ARG (key $LOCAL_KEY)"
  ssh -i "$LOCAL_KEY" -p "$REMOTE_PORT_ARG" \
    -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    "$LOCAL_USER@$LOCAL_HOST" "echo SSH_OK; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1 | head -3"
  exit 0
fi

[[ -n "$PUBKEY" ]] || { echo "ERROR: no pubkey. Pass --pubkey or set PUBKEY/MOLAB_SSH_PUBKEY." >&2; usage; }
[[ "$PUBKEY" == ssh-* ]] || { echo "ERROR: pubkey must start with 'ssh-...'." >&2; exit 1; }

SUDO=""
[[ "$(id -u)" -eq 0 ]] || SUDO="sudo"

# 1. sshd ----------------------------------------------------------------
if ! command -v sshd >/dev/null 2>&1; then
  echo "[up] installing openssh-server..."
  $SUDO apt-get update && $SUDO apt-get install -y openssh-server
fi
$SUDO mkdir -p /run/sshd /var/run/sshd
$SUDO ssh-keygen -A >/dev/null 2>&1 || true

# 2. authorized_keys in passwd-home AND $HOME (they differ on Molab) -----
install_key() {
  local dir="$1"
  mkdir -p "$dir/.ssh"
  chmod 700 "$dir/.ssh"
  touch "$dir/.ssh/authorized_keys"
  chmod 600 "$dir/.ssh/authorized_keys"
  grep -qxF "$PUBKEY" "$dir/.ssh/authorized_keys" 2>/dev/null \
    || echo "$PUBKEY" >> "$dir/.ssh/authorized_keys"
  echo "[up] key installed: $dir/.ssh/authorized_keys"
}
PASSWD_HOME="$(getent passwd "$(whoami)" | cut -d: -f6)"
[[ -n "$PASSWD_HOME" ]] || PASSWD_HOME="$HOME"
install_key "$PASSWD_HOME"
[[ "$HOME" == "$PASSWD_HOME" ]] || install_key "$HOME"

# 3. sshd on $PORT (idempotent: start only if nothing listens) ------------
port_listening() {
  if command -v ss >/dev/null 2>&1; then
    ss -tln 2>/dev/null | grep -q ":$PORT "
  else
    # fallback: hex port in /proc/net/tcp*
    local hex
    hex=$(printf '%04X' "$PORT")
    grep -qi ":$hex " /proc/net/tcp /proc/net/tcp6 2>/dev/null
  fi
}
if port_listening; then
  echo "[up] sshd already listening on $PORT"
else
  echo "[up] starting sshd on $PORT..."
  $SUDO /usr/sbin/sshd -p "$PORT"
  sleep 1
  port_listening && echo "[up] sshd listening on $PORT" \
    || { echo "ERROR: sshd failed to bind $PORT" >&2; exit 1; }
fi

# 4. bore (no account, 443-based; tmate is DNS-blocked on Molab) ----------
if [[ ! -x "$BORE_BIN" ]]; then
  echo "[up] installing bore v$BORE_VER..."
  curl -fsSL "https://github.com/ekzhang/bore/releases/download/v${BORE_VER}/bore-v${BORE_VER}-x86_64-unknown-linux-musl.tar.gz" \
    | tar xz bore
  chmod +x bore
  BORE_BIN="./bore"
fi
pkill -f "bore local $PORT" 2>/dev/null || true
sleep 1
rm -f /tmp/bore.log
nohup "$BORE_BIN" local "$PORT" --to bore.pub > /tmp/bore.log 2>&1 &
echo "[up] bore started, waiting for tunnel..."
for _ in $(seq 1 30); do
  grep -q "listening at" /tmp/bore.log 2>/dev/null && break
  sleep 1
done
grep "listening at" /tmp/bore.log || { echo "ERROR: bore failed:" >&2; cat /tmp/bore.log >&2; exit 1; }

# 5. bundle ---------------------------------------------------------------
REMOTE_PORT="$(sed -n 's/.*listening at bore\.pub:\([0-9]*\).*/\1/p' /tmp/bore.log | tail -1)"
USER_NOW="$(whoami)"
echo
echo "================ MOLAB SSH BUNDLE ================"
echo "BORE_ADDR: bore.pub:$REMOTE_PORT  (forwards to localhost:$PORT on $(hostname))"
echo "SSH_USER:  $USER_NOW"
echo
echo "--- local ~/.ssh/config stanza ---"
cat <<EOF
Host molab-bore
  HostName bore.pub
  Port $REMOTE_PORT
  User $USER_NOW
  IdentityFile ~/.ssh/molab_bore
  StrictHostKeyChecking accept-new
EOF
echo
echo "--- local test ---"
echo "ssh -i ~/.ssh/molab_bore -p $REMOTE_PORT $USER_NOW@bore.pub \"echo SSH_OK; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv | head -3\""
echo "=================================================="
echo "Keep this tab + bore alive. PORT is ephemeral: rerun this script if it changes."
