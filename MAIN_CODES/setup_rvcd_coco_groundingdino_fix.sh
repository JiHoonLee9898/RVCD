#!/usr/bin/env bash
# RVCD + COCO2014 + GroundingDINO one-shot setup/fix script
#
# This script is designed for the issue sequence:
# - unzip not installed
# - pip build isolation cannot see torch
# - nvcc missing / wrong CUDA version
# - g++ too new for CUDA 11.7
# - missing CUDA dev headers: cuda_runtime_api.h, cusparse.h, cublas_v2.h, cusolverDn.h, thrust/complex.h
# - linker cannot find -lcudart
#
# Default paths match Lee JiHoon's server layout:
#   RVCD_MAIN_CODES=/home/jihoon/jihoon/RVCD/MAIN_CODES
#   DATASETS_DIR=/home/jihoon/jihoon/DATASETS
#   CONDA_ENV=RVCD
#
# Usage:
#   bash setup_rvcd_coco_groundingdino_fix.sh
#
# Optional overrides:
#   CONDA_ENV=RVCD \
#   RVCD_MAIN_CODES=/path/to/RVCD/MAIN_CODES \
#   DATASETS_DIR=/path/to/DATASETS \
#   bash setup_rvcd_coco_groundingdino_fix.sh

set -Eeuo pipefail

CONDA_ENV="${CONDA_ENV:-RVCD}"
RVCD_MAIN_CODES="${RVCD_MAIN_CODES:-/home/jihoon/jihoon/RVCD/MAIN_CODES}"
DATASETS_DIR="${DATASETS_DIR:-/home/jihoon/jihoon/DATASETS}"
COCO_ROOT="${COCO_ROOT:-${DATASETS_DIR}/coco2014}"
GROUNDINGDINO_DIR="${GROUNDINGDINO_DIR:-${RVCD_MAIN_CODES}/decoder_zoo/GroundingDINO}"
PYTHON_BIN="${PYTHON_BIN:-python}"

log() {
  echo
  echo "============================================================"
  echo "[RVCD-SETUP] $*"
  echo "============================================================"
}

warn() {
  echo "[WARN] $*" >&2
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

activate_conda_env() {
  log "Activating conda env: ${CONDA_ENV}"

  if have_cmd conda; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    die "conda를 찾지 못했습니다. 먼저 conda/miniconda를 설치하거나, conda init 후 다시 실행하세요."
  fi

  conda activate "${CONDA_ENV}"
  echo "CONDA_PREFIX=${CONDA_PREFIX}"
}

download_file() {
  local url="$1"
  local out="$2"

  if [ -f "$out" ]; then
    echo "Already exists: $out"
    return 0
  fi

  echo "Downloading: $url"
  echo "         to: $out"

  if have_cmd wget; then
    wget -c -O "$out" "$url"
  elif have_cmd curl; then
    curl -L -C - -o "$out" "$url"
  else
    "$PYTHON_BIN" - "$url" "$out" <<'PY'
import sys
import urllib.request

url, out = sys.argv[1], sys.argv[2]
urllib.request.urlretrieve(url, out)
print(f"Downloaded: {out}")
PY
  fi
}

extract_zip_with_python() {
  local zip_path="$1"
  local dest="$2"

  [ -f "$zip_path" ] || die "zip 파일이 없습니다: $zip_path"

  echo "Extracting with python zipfile: $zip_path"
  "$PYTHON_BIN" -m zipfile -e "$zip_path" "$dest"
}

setup_coco2014() {
  log "Setting up COCO2014 under ${COCO_ROOT}"

  mkdir -p "${COCO_ROOT}"
  cd "${COCO_ROOT}"

  download_file "http://images.cocodataset.org/zips/val2014.zip" "val2014.zip"
  download_file "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" "annotations_trainval2014.zip"

  if [ ! -d "${COCO_ROOT}/val2014" ]; then
    extract_zip_with_python "${COCO_ROOT}/val2014.zip" "${COCO_ROOT}"
  else
    echo "Already extracted: ${COCO_ROOT}/val2014"
  fi

  if [ ! -d "${COCO_ROOT}/annotations" ]; then
    extract_zip_with_python "${COCO_ROOT}/annotations_trainval2014.zip" "${COCO_ROOT}"
  else
    echo "Already extracted: ${COCO_ROOT}/annotations"
  fi

  # RVCD code expects:
  #   --data_path/.../COCO_val2014_*.jpg
  #   --data_path/annotations/*.json
  ln -sfn ../annotations "${COCO_ROOT}/val2014/annotations"

  echo
  echo "COCO verification:"
  ls "${COCO_ROOT}/val2014/annotations/captions_val2014.json"
  ls "${COCO_ROOT}/val2014/annotations/instances_val2014.json"

  # This image ID is commonly present in val2014, but if it is not, list any one image.
  if [ -f "${COCO_ROOT}/val2014/COCO_val2014_000000000042.jpg" ]; then
    ls "${COCO_ROOT}/val2014/COCO_val2014_000000000042.jpg"
  else
    find "${COCO_ROOT}/val2014" -maxdepth 1 -name 'COCO_val2014_*.jpg' | head -1
  fi

  echo
  echo "Use this for RVCD:"
  echo "  --data_path ${COCO_ROOT}/val2014"
}

conda_install_try() {
  # Try several conda install commands until one succeeds.
  # Usage:
  #   conda_install_try "description" "conda install ..." "conda install ..."
  local desc="$1"
  shift

  log "Installing/checking: ${desc}"

  local cmd
  for cmd in "$@"; do
    echo "+ ${cmd}"
    if bash -lc "${cmd}"; then
      echo "OK: ${desc}"
      return 0
    fi
    warn "Failed command, trying next fallback: ${cmd}"
  done

  die "설치 실패: ${desc}"
}

setup_cuda_and_compilers() {
  log "Installing CUDA 11.7 build dependencies and GCC/G++ 11"

  # Keep nvcc aligned with torch 2.0.0+cu117.
  conda_install_try "CUDA 11.7 nvcc/runtime/dev headers" \
    "conda install -c nvidia cuda-nvcc=11.7 cuda-cudart=11.7 cuda-cudart-dev=11.7 cuda-libraries-dev=11.7 -y" \
    "conda install -c nvidia/label/cuda-11.7.0 cuda-nvcc cuda-cudart cuda-cudart-dev cuda-libraries-dev -y"

  # CUDA 11.7 requires host g++ < 12. Use conda g++ 11.
  conda_install_try "GCC/G++ 11 for CUDA 11.7" \
    "conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y"

  # CUDA dev headers frequently missing from minimal envs.
  conda_install_try "cuSPARSE dev header: cusparse.h" \
    "conda install -c nvidia libcusparse-dev=11.7.5.86 -y" \
    "conda install -c nvidia/label/cuda-11.7.0 libcusparse-dev=11.7.4.91 -y" \
    "conda install -c nvidia libcusparse-dev -y"

  conda_install_try "cuBLAS dev header: cublas_v2.h" \
    "conda install -c nvidia libcublas-dev=11.10.3.66 -y" \
    "conda install -c nvidia/label/cuda-11.7.0 libcublas-dev=11.10.1.25 -y" \
    "conda install -c nvidia libcublas-dev -y"

  conda_install_try "cuSOLVER dev header: cusolverDn.h" \
    "conda install -c nvidia libcusolver-dev=11.4.0.1 -y" \
    "conda install -c nvidia/label/cuda-11.7.0 libcusolver-dev -y" \
    "conda install -c nvidia libcusolver-dev -y"

  # IMPORTANT: install CUDA 11.7 Thrust from CUDA 11.7 label first to avoid CUDA 13 mixing.
  conda_install_try "Thrust header: thrust/complex.h" \
    "conda install -c nvidia/label/cuda-11.7.0 cuda-thrust -y" \
    "conda install -c nvidia cuda-thrust -y"

  export CUDA_HOME="${CONDA_PREFIX}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
  export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"

  # Fix possible CUDA 13 libcudart symlink pollution.
  # Prefer PyTorch's CUDA 11 runtime if present.
  local pyver
  pyver="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  local torch_cudart="${CONDA_PREFIX}/lib/python${pyver}/site-packages/nvidia/cuda_runtime/lib/libcudart.so.11.0"

  if [ -f "${torch_cudart}" ]; then
    rm -f "${CONDA_PREFIX}/lib/libcudart.so"
    ln -s "${torch_cudart}" "${CONDA_PREFIX}/lib/libcudart.so"
  elif [ -f "${CONDA_PREFIX}/targets/x86_64-linux/lib/libcudart.so" ]; then
    rm -f "${CONDA_PREFIX}/lib/libcudart.so"
    ln -s "${CONDA_PREFIX}/targets/x86_64-linux/lib/libcudart.so" "${CONDA_PREFIX}/lib/libcudart.so"
  fi

  log "Sanity checks"

  echo "Python:"
  which "${PYTHON_BIN}"
  "${PYTHON_BIN}" --version

  echo
  echo "PyTorch:"
  "${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

  echo
  echo "nvcc:"
  which nvcc
  nvcc --version

  echo
  echo "compiler:"
  "${CC}" --version | head -1
  "${CXX}" --version | head -1

  echo
  echo "headers/libraries:"
  ls "${CONDA_PREFIX}/include/cuda_runtime_api.h"
  ls "${CONDA_PREFIX}/include/cusparse.h"
  ls "${CONDA_PREFIX}/include/cublas_v2.h"
  ls "${CONDA_PREFIX}/include/cusolverDn.h"
  ls "${CONDA_PREFIX}/include/thrust/complex.h"
  ls -l "${CONDA_PREFIX}/lib/libcudart.so"
}

write_activation_hook() {
  log "Writing conda activation hook for RVCD build environment"

  mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"

  cat > "${CONDA_PREFIX}/etc/conda/activate.d/rvcd_build_env.sh" <<'EOF'
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
EOF

  cat "${CONDA_PREFIX}/etc/conda/activate.d/rvcd_build_env.sh"
}

build_groundingdino() {
  log "Building GroundingDINO"

  [ -d "${GROUNDINGDINO_DIR}" ] || die "GroundingDINO directory not found: ${GROUNDINGDINO_DIR}"

  cd "${GROUNDINGDINO_DIR}"

  export CUDA_HOME="${CONDA_PREFIX}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
  export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"

  rm -rf build *.egg-info groundingdino.egg-info

  set +e
  MAX_JOBS=1 pip install -e . --no-build-isolation --no-cache-dir -v 2>&1 | tee build_groundingdino.log
  local status=${PIPESTATUS[0]}
  set -e

  if [ "${status}" -ne 0 ]; then
    echo
    echo "GroundingDINO build failed. Key error lines:"
    grep -n -i -E "fatal error|error:|FAILED:|unsupported|killed|No such|undefined|nvcc|sm_|gcc|g\+\+|ld:" build_groundingdino.log | head -160 || true
    die "GroundingDINO 빌드 실패. 위 핵심 에러를 확인하세요: ${GROUNDINGDINO_DIR}/build_groundingdino.log"
  fi
}

verify_groundingdino() {
  log "Verifying GroundingDINO import"

  cd "${RVCD_MAIN_CODES}"

  "${PYTHON_BIN}" - <<'PY'
from groundingdino.util.inference import load_model
print("GroundingDINO import OK")
PY

  echo
  echo "Final CUDA/PyTorch check:"
  "${PYTHON_BIN}" - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

  echo
  echo "nvcc:"
  which nvcc
  nvcc --version | grep "release" || nvcc --version

  echo
  echo "COCO data path for RVCD:"
  echo "  --data_path ${COCO_ROOT}/val2014"

  echo
  echo "DONE."
  echo
  echo "NOTE:"
  echo "  If your GPU is Blackwell / sm_120, torch 2.0.0+cu117 may still warn or fail at runtime."
  echo "  Installation/import can succeed, but actual CUDA kernels may require a newer PyTorch CUDA 12.8+ environment."
}

main() {
  activate_conda_env
  setup_coco2014
  setup_cuda_and_compilers
  write_activation_hook
  build_groundingdino
  verify_groundingdino
}

main "$@"
