#!/usr/bin/env bash
# Fetch the GAOT model code (https://github.com/camlab-ethz/GAOT) for the GAOT
# baseline. Upstream ships no license, so its code is not redistributed here;
# this script copies upstream's `src/model/` at the pinned commit into
# dandelion/models/baselines/gaot/_upstream/ (gitignored).
set -euo pipefail

COMMIT=549c5a5f7113e23ba5e91469f2f8bbb1567fae46
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLONE="${REPO_ROOT}/third_party/GAOT"
DEST="${REPO_ROOT}/dandelion/models/baselines/gaot/_upstream"

if [ ! -d "${CLONE}/.git" ]; then
    git clone https://github.com/camlab-ethz/GAOT.git "${CLONE}"
fi
git -C "${CLONE}" fetch --quiet origin
git -C "${CLONE}" checkout --quiet "${COMMIT}"

rm -rf "${DEST}"
mkdir -p "${DEST}/layers/utils"
cp "${CLONE}/src/model/gaot.py" "${DEST}/"
cp "${CLONE}"/src/model/layers/{agno,attn,gemb,magno,mlp}.py "${DEST}/layers/"
cp "${CLONE}"/src/model/layers/utils/{activation,dataclass,edge_drop,neighbor_search,segment_csr}.py \
    "${DEST}/layers/utils/"
touch "${DEST}/__init__.py" "${DEST}/layers/__init__.py" "${DEST}/layers/utils/__init__.py"
echo "GAOT model code @ ${COMMIT} -> ${DEST}"
