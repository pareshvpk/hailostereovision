#!/usr/bin/env bash
# Install the Hailo Dataflow Compiler into a clean venv and verify it.
#
#     bash deploy/install_dfc.sh ~/Downloads/hailo_dataflow_compiler-*.whl
#
# The DFC is not on PyPI -- it is a login-gated wheel from the Hailo Developer
# Zone (Software Downloads -> Dataflow Compiler). Download it first, then pass
# the path here.
#
# Two things this handles that a bare `pip install` does not:
#
#  1. PYTHONPATH. This machine has a ROS Humble overlay exported into every
#     shell, and those paths sort AHEAD of the venv's site-packages. The DFC
#     pins numpy/protobuf/tensorflow tightly, so the venv is entered with
#     PYTHONPATH cleared -- `hailo-py` below does that for you.
#  2. A clean venv. The DFC must not share an environment with the torch stack
#     in ~/.venvs/stereo; their numpy pins disagree.
set -euo pipefail

WHL="${1:-}"
VENV="$HOME/.venvs/hailo-dfc"

if [[ -z "$WHL" || ! -f "$WHL" ]]; then
    echo "usage: bash deploy/install_dfc.sh <path-to-hailo_dataflow_compiler-*.whl>" >&2
    echo >&2
    echo "Get the wheel from https://hailo.ai/developer-zone/software-downloads/" >&2
    echo >&2
    echo "It is NOT under Vision Processors -> Hailo-15H. That product offers" >&2
    echo "the Vision Processor Software Package, which is the SoC-side BSP." >&2
    echo "The compiler is host-side and device-agnostic:" >&2
    echo >&2
    echo "    Package: Hailo Dataflow Compiler - Python package (whl)" >&2
    echo "    File:    hailo_dataflow_compiler-5.4.0-py3-none-linux_x86_64.whl" >&2
    echo "    Devices: Hailo-10H, Hailo-15H, Hailo-15L" >&2
    echo "    Version: 5.4.0 -- matches the Model Zoo release that built the" >&2
    echo "             stereonet HEF this model is benchmarked against" >&2
    echo >&2
    echo "The same wheel targets hailo15h via --hw-arch; the installer" >&2
    echo "verifies that before finishing." >&2
    exit 1
fi

echo "== apt prerequisites =="
missing=()
for p in build-essential python3-dev graphviz libgraphviz-dev python3.10-venv; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
done
if (( ${#missing[@]} )); then
    echo "installing: ${missing[*]}"
    sudo apt-get install -y "${missing[@]}"
else
    echo "all present"
fi

echo "== venv =="
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
env -u PYTHONPATH "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

echo "== installing $(basename "$WHL") =="
env -u PYTHONPATH "$VENV/bin/python" -m pip install "$WHL"

# Nothing extra to install: the 5.4.0 wheel already requires Pillow (used by
# dfc_flow.py's KITTI loader) alongside numpy==1.26.4, tensorflow==2.19.1,
# protobuf==3.20.3, onnx==1.17.0 and pygraphviz. That protobuf pin is why the
# ROS PYTHONPATH must stay cleared, and pygraphviz is why libgraphviz-dev is
# not optional -- it builds from source against those headers.

echo "== verify =="
env -u PYTHONPATH "$VENV/bin/python" - <<'PY'
import hailo_sdk_client as c
print("hailo_sdk_client", getattr(c, "__version__", "?"))
from hailo_sdk_client import ClientRunner, InferenceContext

# The DFC is one device-agnostic tool; the target is a --hw-arch flag, not a
# separate download. The Developer Zone only offers it under Accelerators, so
# it is reached by selecting a Hailo-8/10H device -- that says nothing about
# which architectures the wheel can compile for. Prove hailo15h is in there.
archs = ["hailo8", "hailo8l", "hailo10h", "hailo15h", "hailo15l"]
supported = []
for a in archs:
    try:
        ClientRunner(hw_arch=a)
        supported.append(a)
    except Exception:
        pass
print("supported architectures:", ", ".join(supported) or "NONE")
if "hailo15h" not in supported:
    raise SystemExit(
        "\nFAIL: this DFC build does not support hailo15h.\n"
        "It is too old -- the Model Zoo has shipped 15H HEFs since the v5.x\n"
        "compiled releases. Download a newer Dataflow Compiler.")
print("hailo15h: OK -- this is the target dfc_flow.py compiles for")
print("inference contexts:", [x.name for x in InferenceContext])
import numpy, PIL
print("numpy", numpy.__version__, "| pillow", PIL.__version__)
PY

cat <<EOF

== done ==
The DFC lives in $VENV and must be run with PYTHONPATH cleared.
A wrapper is installed at deploy/hailo-py:

    ./deploy/hailo-py deploy/dfc_flow.py all

EOF
