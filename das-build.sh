#!/usr/bin/env bash
# i12055: bounded CI environment hold for the HCU runner probe (temporary).
set -uo pipefail
echo "I12055 tile-ai hold begin $(date -u +%Y-%m-%dT%H:%M:%SZ) runner=${RUNNER_NAME:-?} job=${GITHUB_JOB:-?} run=${GITHUB_RUN_ID:-?}"
sleep 720
echo "I12055 tile-ai hold end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
