#!/usr/bin/env bash
# Re-run a command until it succeeds. The GPU is shared; a transient CUDA out-of-memory error only
# costs the model that was being fitted because finished models are cached under results/cache/.
# usage: scripts/retry.sh <max_attempts> <command...>
max=$1; shift
for i in $(seq 1 "$max"); do
  "$@" && exit 0
  echo "[retry] attempt $i failed; sleeping 60s" >&2
  sleep 60
done
exit 1
