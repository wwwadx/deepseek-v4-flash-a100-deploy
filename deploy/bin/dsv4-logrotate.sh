#!/usr/bin/env bash
# Size-based rotation for the DeepSeek-V4-Flash server log.
# logrotate is NOT installed on this host, hence this standalone rotator.
# systemd holds an append fd on the log, so we MUST copy-then-truncate:
# renaming would leave the server writing into an unlinked inode.
set -euo pipefail
LOG=/data1/dsv4/v2_attempt/serve_final.log
MAX=209715200          # 200 MB
KEEP=8
[ -f "$LOG" ] || exit 0
SIZE=$(stat -c %s "$LOG")
[ "$SIZE" -lt "$MAX" ] && exit 0
for i in $(seq $((KEEP-1)) -1 1); do
  [ -f "$LOG.$i.gz" ] && mv -f "$LOG.$i.gz" "$LOG.$((i+1)).gz"
done
cp -a "$LOG" "$LOG.1"
: > "$LOG"                      # truncate in place, fd stays valid
gzip -f "$LOG.1"
chmod 640 "$LOG" "$LOG.1.gz" 2>/dev/null || true
rm -f "$LOG.$((KEEP+1)).gz"
logger -t dsv4-logrotate "rotated serve_final.log at ${SIZE} bytes"
