#!/bin/bash
# One-time environment setup on a PBS cluster (run on the login node):
#   bash cluster/setup.sh
# Our cluster runs an old glibc (2.17), so compiled packages come from
# conda-forge via micromamba and only pure-Python packages from pip.
set -eu

ENV_NAME=${ENV_NAME:-atcrl}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
PREFIX=$CONDA_ROOT/envs/$ENV_NAME
PY=$PREFIX/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)

MAMBA=$(command -v micromamba || true)
if [ -z "$MAMBA" ]; then
    MAMBA=$HOME/.local/micromamba/bin/micromamba
    if [ ! -x "$MAMBA" ]; then
        mkdir -p "$HOME/.local/micromamba"
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
            | tar -xj -C "$HOME/.local/micromamba" bin/micromamba
    fi
fi

"$MAMBA" create -y -p "$PREFIX" -c conda-forge --override-channels \
    python=3.12 pip "numpy>=1.26,<2.4" scipy matplotlib-base shapely pygame pillow
"$PY" -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu torch

# keep pip from replacing the conda numpy
NUMPY=$("$PY" -c "import numpy; print(numpy.__version__)")
"$PY" -m pip install --only-binary=:all: "numpy==$NUMPY" -r "$HERE/requirements.txt" || {
    # bluesky-simulator pulls a Qt wheel that does not exist for glibc 2.17;
    # a detached simulation never imports Qt, so install it without deps
    "$PY" -m pip install --only-binary=:all: "numpy==$NUMPY" \
        gymnasium stable-baselines3 pettingzoo supersuit msgpack pyzmq openap bluesky-navdata
    "$PY" -m pip install --no-deps bluesky-simulator
}

"$PY" -c "import torch, stable_baselines3, bluesky as bs; bs.init(mode='sim', detached=True); print('environment ready')"
