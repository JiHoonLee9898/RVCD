#!/usr/bin/env bash
# setup_rvcd_reliable_a100_blackwell_fixed_v2.sh
#
# RVCD reliable all-in-one setup for:
#   - A100 / Ampere / Hopper-class NVIDIA GPUs
#   - Blackwell GPUs
#
# Main fixes vs previous script:
#   1. A100/base path uses CUDA 11.8 instead of CUDA 11.7.
#   2. A100/base path no longer pins old bitsandbytes==0.37.0.
#   3. bitsandbytes is installed/tested separately with BNB_CUDA_VERSION/CUDA_VERSION forced.
#   4. bitsandbytes import failure is non-fatal by default; set STRICT_BNB=1 to fail hard.
#   5. Removed the erroneous unconditional "conda activate RVCD_BW" after main().
#   6. PatternLite/OpenJDK install is moved into an optional function and uses the detected env.
#   7. RVCD_ROOT defaults to the current script directory, with fallbacks for common paths.
#
# Usage:
#   cd /path/to/RVCD
#   bash setup_rvcd_reliable_a100_blackwell_fixed_v2.sh
#
# Useful options:
#   INSTALL_COCO=0 bash setup_rvcd_reliable_a100_blackwell_fixed_v2.sh
#   INSTALL_WEIGHTS=0 bash setup_rvcd_reliable_a100_blackwell_fixed_v2.sh
#   INSTALL_PATTERNLITE=0 bash setup_rvcd_reliable_a100_blackwell_fixed_v2.sh
#   STRICT_BNB=1 bash setup_rvcd_reliable_a100_blackwell_fixed_v2.sh
#   RVCD_ROOT=/home/jihoon/jihoon/RVCD bash setup_rvcd_reliable_a100_blackwell_fixed_v2.sh
#
# If anything fails, logs are written under:
#   ${RVCD_ROOT}/setup_logs

set -Eeo pipefail

# ----------------------------
# Paths
# ----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${RVCD_ROOT:-}" ]; then
  if [ -d "${SCRIPT_DIR}/MAIN_CODES" ]; then
    RVCD_ROOT="${SCRIPT_DIR}"
  elif [ -d "/home/jihoon/jihoon/RVCD/MAIN_CODES" ]; then
    RVCD_ROOT="/home/jihoon/jihoon/RVCD"
  elif [ -d "/root/RVCD/MAIN_CODES" ]; then
    RVCD_ROOT="/root/RVCD"
  else
    RVCD_ROOT="${SCRIPT_DIR}"
  fi
fi

RVCD_MAIN_CODES="${RVCD_MAIN_CODES:-${RVCD_ROOT}/MAIN_CODES}"
GROUNDINGDINO_DIR="${GROUNDINGDINO_DIR:-${RVCD_MAIN_CODES}/decoder_zoo/GroundingDINO}"

DATASETS_DIR="${DATASETS_DIR:-/root/DATASETS}"
COCO_ROOT="${COCO_ROOT:-${DATASETS_DIR}/coco2014}"
DATA_PATH="${DATA_PATH:-${COCO_ROOT}/val2014}"

TRANSFORMERS_DIR="${TRANSFORMERS_DIR:-${RVCD_ROOT}/transformers-4.36.2}"
TRANSFORMERS_ALT_DIR="${TRANSFORMERS_ALT_DIR:-${RVCD_MAIN_CODES}/transformers-4.36.2}"

INSTALL_COCO="${INSTALL_COCO:-1}"
INSTALL_WEIGHTS="${INSTALL_WEIGHTS:-1}"
INSTALL_PATTERNLITE="${INSTALL_PATTERNLITE:-1}"
INSTALL_PATTERN_GIT="${INSTALL_PATTERN_GIT:-0}"

WEIGHTS_URL="${WEIGHTS_URL:-https://drive.google.com/drive/folders/1UaMJga-BKju88CXAdonbiQujBKkdcVGX}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${RVCD_MAIN_CODES}/decoder_zoo/weights}"

LOG_DIR="${LOG_DIR:-${RVCD_ROOT}/setup_logs}"

# bitsandbytes policy:
#   STRICT_BNB=0: warn and continue if bitsandbytes fails.
#   STRICT_BNB=1: fail setup if bitsandbytes fails.
STRICT_BNB="${STRICT_BNB:-0}"

# ----------------------------
# GPU-dependent stacks
# ----------------------------
# Blackwell path
BW_ENV="${BW_ENV:-RVCD_BW}"
BW_PYTHON="${BW_PYTHON:-3.10}"
BW_TORCH="${BW_TORCH:-2.7.1}"
BW_TORCHVISION="${BW_TORCHVISION:-0.22.1}"
BW_TORCHAUDIO="${BW_TORCHAUDIO:-2.7.1}"
BW_CUDA="${BW_CUDA:-12.8}"
BW_ARCH_LIST="${BW_ARCH_LIST:-12.0}"
BW_BNB_PACKAGE="${BW_BNB_PACKAGE:-bitsandbytes>=0.46.0}"

# Base/A100 path
#
# Important:
#   The older combination torch cu117 + bitsandbytes==0.37.0 often misdetects
#   modern driver/container CUDA as 13.0 and crashes on import.
#
#   This base path intentionally uses:
#     - torch 2.0.0 cu118
#     - CUDA 11.8 build tools
#     - bitsandbytes 0.42.0
#     - BNB_CUDA_VERSION/CUDA_VERSION=118
#
#   This keeps the RVCD-era torch stack close to the original script while avoiding
#   the CUDA 13.0 bitsandbytes autodetection trap on A100.
BASE_ENV="${BASE_ENV:-RVCD}"
BASE_PYTHON="${BASE_PYTHON:-3.9}"
BASE_TORCH="${BASE_TORCH:-2.0.0}"
BASE_TORCHVISION="${BASE_TORCHVISION:-0.15.1}"
BASE_TORCHAUDIO="${BASE_TORCHAUDIO:-2.0.1}"
BASE_CUDA="${BASE_CUDA:-11.8}"
BASE_BNB_PACKAGE="${BASE_BNB_PACKAGE:-bitsandbytes==0.42.0}"

# ----------------------------
# Helpers
# ----------------------------
log() {
  echo
  echo "============================================================"
  echo "[RVCD_RELIABLE_SETUP] $*"
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

init_logs() {
  mkdir -p "${LOG_DIR}"
  echo "Logs will be saved under: ${LOG_DIR}"
}

run_logged() {
  # Usage:
  #   run_logged log_filename command args...
  local logfile="$1"
  shift
  echo "+ $*" | tee -a "${LOG_DIR}/${logfile}"
  set +e
  "$@" 2>&1 | tee -a "${LOG_DIR}/${logfile}"
  local status=${PIPESTATUS[0]}
  set -e
  if [ "${status}" -ne 0 ]; then
    echo
    echo "[ERROR] Command failed. Log: ${LOG_DIR}/${logfile}" >&2
    grep -n -i -E "error|failed|conflict|resolution|No matching|Could not|subprocess|metadata|wheel|traceback|exception|undefined|CUDA|bitsandbytes" "${LOG_DIR}/${logfile}" | tail -120 || true
    exit "${status}"
  fi
}

run_logged_allow_fail() {
  # Usage:
  #   run_logged_allow_fail log_filename command args...
  local logfile="$1"
  shift
  echo "+ $*" | tee -a "${LOG_DIR}/${logfile}"
  set +e
  "$@" 2>&1 | tee -a "${LOG_DIR}/${logfile}"
  local status=${PIPESTATUS[0]}
  set -e
  return "${status}"
}

prepend_path_dir() {
  local d="$1"
  if [ -d "${d}" ]; then
    case ":${PATH}:" in
      *":${d}:"*) ;;
      *) export PATH="${d}:${PATH}" ;;
    esac
  fi
}

prepend_ld_dir() {
  local d="$1"
  if [ -d "${d}" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":${d}:"*) ;;
      *) export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}" ;;
    esac
  fi
}

cuda_code_from_version() {
  # "11.8" -> "118", "12.8" -> "128"
  echo "$1" | tr -d '.'
}

setup_cuda_library_path() {
  # Make CUDA runtime libraries from pip torch / nvidia wheels discoverable.
  # This is especially important for bitsandbytes import checks.
  local site_pkgs=""
  site_pkgs="$(python - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"

  prepend_ld_dir "${CONDA_PREFIX}/lib"
  prepend_ld_dir "${CONDA_PREFIX}/targets/x86_64-linux/lib"

  if [ -n "${site_pkgs}" ]; then
    prepend_ld_dir "${site_pkgs}/torch/lib"
    prepend_ld_dir "${site_pkgs}/nvidia/cuda_runtime/lib"
    prepend_ld_dir "${site_pkgs}/nvidia/cublas/lib"
    prepend_ld_dir "${site_pkgs}/nvidia/cufft/lib"
    prepend_ld_dir "${site_pkgs}/nvidia/curand/lib"
    prepend_ld_dir "${site_pkgs}/nvidia/cusolver/lib"
    prepend_ld_dir "${site_pkgs}/nvidia/cusparse/lib"
    prepend_ld_dir "${site_pkgs}/nvidia/nccl/lib"
    prepend_ld_dir "${site_pkgs}/nvidia/nvtx/lib"
  fi

  export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"

  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
}

load_conda() {
  log "Loading conda"

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
    die "conda를 찾지 못했습니다. conda init 후 다시 실행하세요."
  fi

  conda --version
}

detect_gpu0() {
  log "Detecting GPU 0"

  if ! have_cmd nvidia-smi; then
    die "nvidia-smi를 찾지 못했습니다. NVIDIA GPU 서버에서 실행하세요."
  fi

  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/^ *//;s/ *$//')"
  GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 | sed 's/^ *//;s/ *$//' || true)"

  if [ -z "${GPU_NAME}" ]; then
    GPU_NAME="$(nvidia-smi -L | head -n 1 | sed 's/^GPU 0: //;s/ (UUID:.*$//')"
  fi

  echo "GPU 0 name: ${GPU_NAME}"
  echo "GPU 0 compute capability: ${GPU_CC:-unknown}"

  IS_BLACKWELL=0

  if [ -n "${GPU_CC:-}" ]; then
    GPU_CC_MAJOR="${GPU_CC%%.*}"
    if [[ "${GPU_CC_MAJOR}" =~ ^[0-9]+$ ]] && [ "${GPU_CC_MAJOR}" -ge 12 ]; then
      IS_BLACKWELL=1
    fi
  fi

  if echo "${GPU_NAME}" | grep -Eiq 'Blackwell|RTX PRO 6000|RTX 50|RTX 5090|RTX 5080|RTX 5070|RTX 5060|B200|GB200|GB10'; then
    IS_BLACKWELL=1
  fi

  if [ "${IS_BLACKWELL}" -eq 1 ]; then
    MODE="blackwell"
    ENV_NAME="${BW_ENV}"
    PYTHON_VERSION="${BW_PYTHON}"
    TARGET_CUDA="${BW_CUDA}"
    TORCH_VERSION="${BW_TORCH}"
    TORCHVISION_VERSION="${BW_TORCHVISION}"
    TORCHAUDIO_VERSION="${BW_TORCHAUDIO}"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    TARGET_ARCH_LIST="${BW_ARCH_LIST}"
    BNB_PACKAGE="${BW_BNB_PACKAGE}"
  else
    MODE="base"
    ENV_NAME="${BASE_ENV}"
    PYTHON_VERSION="${BASE_PYTHON}"
    TARGET_CUDA="${BASE_CUDA}"
    TORCH_VERSION="${BASE_TORCH}"
    TORCHVISION_VERSION="${BASE_TORCHVISION}"
    TORCHAUDIO_VERSION="${BASE_TORCHAUDIO}"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"

    # For A100 this should become 8.0.
    # For RTX 3090/4090/H100 this uses the actual detected CC where available.
    TARGET_ARCH_LIST="8.0"
    if [ -n "${GPU_CC:-}" ] && [[ "${GPU_CC}" =~ ^[0-9]+\.[0-9]+$ ]]; then
      TARGET_ARCH_LIST="${GPU_CC}"
    fi

    BNB_PACKAGE="${BASE_BNB_PACKAGE}"
  fi

  BNB_CUDA_CODE="$(cuda_code_from_version "${TARGET_CUDA}")"

  cat <<EOF
Selected setup:
  MODE=${MODE}
  ENV_NAME=${ENV_NAME}
  PYTHON_VERSION=${PYTHON_VERSION}
  TORCH_VERSION=${TORCH_VERSION}
  TORCHVISION_VERSION=${TORCHVISION_VERSION}
  TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}
  TORCH_INDEX_URL=${TORCH_INDEX_URL}
  TARGET_CUDA=${TARGET_CUDA}
  TORCH_CUDA_ARCH_LIST=${TARGET_ARCH_LIST}
  BNB_PACKAGE=${BNB_PACKAGE}
  BNB_CUDA_VERSION=${BNB_CUDA_CODE}
  STRICT_BNB=${STRICT_BNB}
EOF
}

safe_deactivate_target() {
  set +u 2>/dev/null || true

  local n=0
  while [ "${CONDA_DEFAULT_ENV:-}" = "${ENV_NAME}" ] && [ "${n}" -lt 8 ]; do
    echo "Currently inside ${ENV_NAME}; deactivating before removal..."
    conda deactivate || true
    n=$((n + 1))
  done

  if [ "${CONDA_DEFAULT_ENV:-}" = "${ENV_NAME}" ]; then
    die "아직 ${ENV_NAME} 안입니다. 새 터미널에서 base 상태로 실행하거나 conda deactivate 후 다시 실행하세요."
  fi
}

create_clean_env() {
  log "Always deleting and recreating env: ${ENV_NAME}"

  safe_deactivate_target

  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    run_logged "conda_env_remove.log" conda env remove -n "${ENV_NAME}" -y
  fi

  run_logged "conda_env_create.log" conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
  conda activate "${ENV_NAME}"

  echo "Activated: ${CONDA_DEFAULT_ENV}"
  echo "CONDA_PREFIX=${CONDA_PREFIX}"
  python --version
}

install_torch_stack() {
  log "Installing GPU-compatible torch stack first"

  run_logged "pip_basics.log" python -m pip install -U "pip<26" "setuptools<81" wheel

  run_logged "pip_torch.log" python -m pip install --no-cache-dir \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

  setup_cuda_library_path
  verify_torch_cuda
}

verify_torch_cuda() {
  log "Verifying torch CUDA"

  python - <<PY
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if str(torch.version.cuda) != "${TARGET_CUDA}":
    raise SystemExit(f"Expected torch CUDA ${TARGET_CUDA}, got {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(2, 2, device="cuda")
print("CUDA op OK:", x.mean().item())
PY
}

ensure_transformers_source() {
  log "Ensuring local ./transformers-4.36.2 source"

  if [ -d "${RVCD_ROOT}/transformers-4.36.2" ]; then
    TRANSFORMERS_DIR="${RVCD_ROOT}/transformers-4.36.2"
    echo "Using existing: ${TRANSFORMERS_DIR}"
    return 0
  fi

  if [ -d "${RVCD_MAIN_CODES}/transformers-4.36.2" ]; then
    TRANSFORMERS_DIR="${RVCD_MAIN_CODES}/transformers-4.36.2"
    echo "Using existing: ${TRANSFORMERS_DIR}"
    return 0
  fi

  TRANSFORMERS_DIR="${RVCD_ROOT}/transformers-4.36.2"
  echo "Missing transformers-4.36.2. Downloading to ${TRANSFORMERS_DIR}"

  if have_cmd git; then
    set +e
    git clone --depth 1 --branch v4.36.2 https://github.com/huggingface/transformers.git "${TRANSFORMERS_DIR}"
    local status=$?
    set -e
    if [ "${status}" -eq 0 ]; then
      return 0
    fi
    rm -rf "${TRANSFORMERS_DIR}"
    warn "git clone failed. Falling back to zip download."
  fi

  python - "${TRANSFORMERS_DIR}" <<'PY'
import sys, urllib.request, zipfile, tempfile, shutil
from pathlib import Path

target = Path(sys.argv[1]).resolve()
url = "https://github.com/huggingface/transformers/archive/refs/tags/v4.36.2.zip"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    z = td / "transformers-v4.36.2.zip"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, z)
    with zipfile.ZipFile(z) as f:
        f.extractall(td)

    src = td / "transformers-4.36.2"
    if not src.exists():
        candidates = list(td.glob("transformers-*"))
        if not candidates:
            raise SystemExit("No extracted transformers directory found")
        src = candidates[0]

    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(src), str(target))

print(f"Ready: {target}")
PY
}

install_transformers_explicitly() {
  log "Installing transformers explicitly"

  cd "${RVCD_ROOT}"
  ensure_transformers_source

  if [ -d "${RVCD_ROOT}/transformers-4.36.2" ]; then
    run_logged "pip_transformers_editable.log" python -m pip install -e "${RVCD_ROOT}/transformers-4.36.2"
  elif [ -d "${RVCD_MAIN_CODES}/transformers-4.36.2" ]; then
    run_logged "pip_transformers_editable.log" python -m pip install -e "${RVCD_MAIN_CODES}/transformers-4.36.2"
  else
    run_logged "pip_transformers.log" python -m pip install "transformers==4.36.2"
  fi

  python - <<'PY'
import transformers
from transformers import AutoTokenizer
print("transformers OK:", transformers.__version__)
print("AutoTokenizer OK")
PY
}

write_activation_hook() {
  log "Writing persistent PYTHONPATH, CUDA, and bitsandbytes env"

  mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"

  cat > "${CONDA_PREFIX}/etc/conda/activate.d/rvcd_env.sh" <<EOF
export CUDA_HOME=\$CONDA_PREFIX
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CONDA_PREFIX/targets/x86_64-linux/lib:\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=\$CONDA_PREFIX/targets/x86_64-linux/lib:\$CONDA_PREFIX/lib:\${LIBRARY_PATH:-}
export CC=\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export TORCH_CUDA_ARCH_LIST="${TARGET_ARCH_LIST}"
export BNB_CUDA_VERSION="${BNB_CUDA_CODE}"
export CUDA_VERSION="${BNB_CUDA_CODE}"
export PYTHONPATH=${RVCD_MAIN_CODES}:\${PYTHONPATH:-}
EOF

  export CUDA_HOME="${CONDA_PREFIX}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
  export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
  export TORCH_CUDA_ARCH_LIST="${TARGET_ARCH_LIST}"
  export BNB_CUDA_VERSION="${BNB_CUDA_CODE}"
  export CUDA_VERSION="${BNB_CUDA_CODE}"
  export PYTHONPATH="${RVCD_MAIN_CODES}:${PYTHONPATH:-}"

  setup_cuda_library_path

  echo "PYTHONPATH=${PYTHONPATH}"
  echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  echo "BNB_CUDA_VERSION=${BNB_CUDA_VERSION}"
  echo "CUDA_VERSION=${CUDA_VERSION}"
}

install_bitsandbytes_safely() {
  log "Installing bitsandbytes safely"

  write_activation_hook

  # Remove any previously misdetected/broken bnb first.
  python -m pip uninstall -y bitsandbytes >/dev/null 2>&1 || true

  run_logged "pip_bitsandbytes.log" env \
    BNB_CUDA_VERSION="${BNB_CUDA_CODE}" \
    CUDA_VERSION="${BNB_CUDA_CODE}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    python -m pip install --no-cache-dir "${BNB_PACKAGE}"

  set +e
  env \
    BNB_CUDA_VERSION="${BNB_CUDA_CODE}" \
    CUDA_VERSION="${BNB_CUDA_CODE}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    python - <<'PY' 2>&1 | tee "${LOG_DIR}/verify_bitsandbytes.log"
import os
print("BNB_CUDA_VERSION:", os.environ.get("BNB_CUDA_VERSION"))
print("CUDA_VERSION:", os.environ.get("CUDA_VERSION"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "")[:1000])

import torch
print("torch:", torch.__version__, "torch cuda:", torch.version.cuda, "cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

import bitsandbytes as bnb
print("bitsandbytes OK:", getattr(bnb, "__version__", "unknown"))
PY
  local status=${PIPESTATUS[0]}
  set -e

  if [ "${status}" -ne 0 ]; then
    echo
    warn "bitsandbytes import failed. Log: ${LOG_DIR}/verify_bitsandbytes.log"
    warn "This setup will continue because STRICT_BNB=${STRICT_BNB}."
    warn "If your RVCD run actually needs 8-bit/4-bit quantization, rerun with STRICT_BNB=1 after checking the log."
    if [ "${STRICT_BNB}" = "1" ]; then
      die "bitsandbytes failed and STRICT_BNB=1"
    fi
  else
    echo "bitsandbytes import OK"
  fi
}

install_runtime_deps_first() {
  log "Installing RVCD runtime dependencies BEFORE CUDA/GroundingDINO build"

  # This requirements file is saved under the repo, not /tmp, so it remains debuggable.
  REQ_FILE="${LOG_DIR}/rvcd_full_deps_no_torch_no_bnb_runtime.txt"

  run_logged "pip_basics_again.log" python -m pip install -U "pip<26" "setuptools<81" wheel
  run_logged "pip_numpy.log" python -m pip install "numpy==1.26.4"

  log "Installing spaCy and en_core_web_sm 3.5.0"

  run_logged "pip_spacy.log" python -m pip install "spacy==3.5.4"
  run_logged "pip_spacy_model.log" python -m pip install \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl"

  cat > "${REQ_FILE}" <<REQ
huggingface-hub==0.20.2
matplotlib==3.7.0
psutil==5.9.4
iopath
pyyaml==6.0
regex==2022.10.31
tokenizers==0.15.0
tqdm==4.64.1
timm==0.6.13
webdataset==0.2.48
omegaconf==2.3.0
opencv-python==4.7.0.72
decord==0.6.0
peft==0.2.0
sentence-transformers
gradio==3.47.1
accelerate==0.26.1
scikit-image
visual-genome
wandb
seaborn
pandas
shortuuid
chardet
openai
supervision
addict
yapf
pycocotools
pycocoevalcap
icecream==2.1.3
pydantic==1.10.9
hpsv2==1.2.0
ninja
nltk==3.9.1
ultralytics==8.3.27
gdown
REQ

  echo "Saved requirements to: ${REQ_FILE}"
  grep -n "tqdm" "${REQ_FILE}"

  # Install all runtime dependencies except torch and bitsandbytes.
  # bitsandbytes is installed separately so CUDA detection can be controlled.
  run_logged "pip_runtime_deps.log" python -m pip install -r "${REQ_FILE}"

  install_transformers_explicitly

  # Re-pin key packages after sentence-transformers/openai/etc may touch transitive deps.
  run_logged "pip_repin.log" python -m pip install \
    "huggingface-hub==0.20.2" \
    "transformers==4.36.2" \
    "tokenizers==0.15.0" \
    "pydantic==1.10.9" \
    "opencv-python==4.7.0.72" \
    "timm==0.6.13" \
    "ultralytics==8.3.27" \
    "tqdm==4.64.1"

  # Prefer local editable transformers one final time.
  if [ -d "${RVCD_ROOT}/transformers-4.36.2" ]; then
    run_logged "pip_transformers_editable_final.log" python -m pip install -e "${RVCD_ROOT}/transformers-4.36.2"
  elif [ -d "${RVCD_MAIN_CODES}/transformers-4.36.2" ]; then
    run_logged "pip_transformers_editable_final.log" python -m pip install -e "${RVCD_MAIN_CODES}/transformers-4.36.2"
  fi

  verify_torch_cuda
  install_bitsandbytes_safely
  verify_runtime_imports
}

verify_runtime_imports() {
  log "Verifying runtime imports BEFORE CUDA build"

  cd "${RVCD_MAIN_CODES}"
  export PYTHONPATH="${RVCD_MAIN_CODES}:${PYTHONPATH:-}"
  setup_cuda_library_path

  python - <<PY
import importlib
import os
import torch

print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
assert str(torch.version.cuda) == "${TARGET_CUDA}"
assert torch.cuda.is_available()

mods = [
    ("tqdm", "tqdm"),
    ("transformers", "transformers"),
    ("ultralytics", "ultralytics"),
    ("minigpt4", "minigpt4"),
    ("cv2", "cv2"),
    ("timm", "timm"),
    ("omegaconf", "omegaconf"),
    ("pycocotools", "pycocotools"),
    ("pycocoevalcap", "pycocoevalcap"),
    ("spacy", "spacy"),
    ("nltk", "nltk"),
]

failed = []
for label, module in mods:
    try:
        mod = importlib.import_module(module)
        if label == "tqdm":
            print("tqdm version:", mod.__version__)
        print(f"{label} OK")
    except Exception as e:
        failed.append((label, repr(e)))

try:
    from transformers import AutoTokenizer
    print("from transformers import AutoTokenizer OK")
except Exception as e:
    failed.append(("AutoTokenizer", repr(e)))

try:
    from ultralytics import YOLO
    print("from ultralytics import YOLO OK")
except Exception as e:
    failed.append(("YOLO", repr(e)))

try:
    import spacy
    spacy.load("en_core_web_sm")
    print("spaCy en_core_web_sm OK")
except Exception as e:
    failed.append(("spaCy model", repr(e)))

# bitsandbytes is useful for quantization, but many RVCD paths do not require it.
# Keep it non-fatal unless STRICT_BNB=1.
try:
    import bitsandbytes as bnb
    print("bitsandbytes OK:", getattr(bnb, "__version__", "unknown"))
except Exception as e:
    msg = repr(e)
    print("bitsandbytes WARN:", msg)
    if "${STRICT_BNB}" == "1":
        failed.append(("bitsandbytes", msg))

if failed:
    print("\\nFAILED RUNTIME IMPORTS:")
    for label, err in failed:
        print(f"  {label}: {err}")
    raise SystemExit(1)

print("\\nRuntime deps OK before CUDA/GroundingDINO build")
PY
}

install_cuda_build_tools_after_deps() {
  log "Installing CUDA ${TARGET_CUDA} build tools AFTER runtime deps"

  if [ "${MODE}" = "blackwell" ]; then
    run_logged "conda_cuda_build_tools.log" conda install -y \
      -c nvidia/label/cuda-12.8.0 \
      -c nvidia \
      "cuda-nvcc>=12.8,<12.9" \
      "cuda-cudart>=12.8,<12.9" \
      "cuda-cudart-dev>=12.8,<12.9" \
      "cuda-libraries-dev>=12.8,<12.9" \
      "cuda-thrust>=12.8,<12.9"
  else
    run_logged "conda_cuda_build_tools.log" conda install -y \
      -c nvidia/label/cuda-11.8.0 \
      -c nvidia \
      "cuda-nvcc>=11.8,<11.9" \
      "cuda-cudart>=11.8,<11.9" \
      "cuda-cudart-dev>=11.8,<11.9" \
      "cuda-libraries-dev>=11.8,<11.9" \
      "cuda-thrust>=11.8,<11.9"
  fi

  run_logged "conda_compilers.log" conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11 ninja git pyyaml

  write_activation_hook

  which nvcc
  nvcc --version
  if ! nvcc --version | grep -q "release ${TARGET_CUDA}"; then
    conda list | grep -E 'cuda|cublas|cusolver|cusparse|cudart|thrust' || true
    die "nvcc가 CUDA ${TARGET_CUDA}가 아닙니다."
  fi

  "${CC}" --version | head -1
  "${CXX}" --version | head -1

  ls "${CONDA_PREFIX}/include/cuda_runtime_api.h"
  ls "${CONDA_PREFIX}/include/cusparse.h"
  ls "${CONDA_PREFIX}/include/cublas_v2.h"
  ls "${CONDA_PREFIX}/include/cusolverDn.h"
  ls "${CONDA_PREFIX}/include/thrust/complex.h"
}

download_file() {
  local url="$1"
  local out="$2"

  if have_cmd wget; then
    wget -c -O "$out" "$url"
  elif have_cmd curl; then
    curl -L -C - -o "$out" "$url"
  else
    python - "$url" "$out" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
  fi
}

zip_is_valid() {
  local z="$1"
  [ -f "${z}" ] || return 1

  python - "${z}" <<'PY'
import sys, zipfile
from pathlib import Path
z = Path(sys.argv[1])
try:
    with zipfile.ZipFile(z, "r") as f:
        bad = f.testzip()
        if bad:
            print(f"BAD_MEMBER={bad}")
            raise SystemExit(1)
        if not f.namelist():
            print("EMPTY_ZIP")
            raise SystemExit(1)
except Exception as e:
    print(f"ZIP_INVALID: {e}")
    raise SystemExit(1)
print(f"ZIP_OK: {z} ({z.stat().st_size} bytes)")
PY
}

ensure_zip_downloaded_and_valid() {
  local url="$1"
  local out="$2"

  if [ -f "${out}" ]; then
    echo "Found existing zip: ${out}"
    if zip_is_valid "${out}"; then
      return 0
    fi
    warn "Existing zip is corrupt/incomplete. Removing: ${out}"
    rm -f "${out}"
  fi

  echo "Downloading: ${url}"
  download_file "${url}" "${out}"

  if zip_is_valid "${out}"; then
    return 0
  fi

  warn "Downloaded zip failed validation. Retrying once: ${out}"
  rm -f "${out}"
  download_file "${url}" "${out}"
  zip_is_valid "${out}" || die "Zip validation failed after retry: ${out}"
}

extract_zip_fresh() {
  local z="$1"
  local dest="$2"
  local expected_dir="$3"

  echo "Extracting fresh: ${z} -> ${dest}"
  rm -rf "${dest}/${expected_dir}"
  python -m zipfile -e "${z}" "${dest}"
}

repair_coco_symlink() {
  local root="$1"
  local val_dir="${root}/val2014"
  local ann_dir="${root}/annotations"

  [ -d "${val_dir}" ] || return 1
  [ -d "${ann_dir}" ] || return 1

  # Always replace stale/broken/wrong val2014/annotations with the correct link.
  # Correct layout:
  #   ${root}/val2014/annotations -> ../annotations
  rm -rf "${val_dir}/annotations"
  ln -sfn ../annotations "${val_dir}/annotations"
}

coco_ready_at() {
  local candidate="$1"
  local root=""

  [ -d "${candidate}" ] || return 1
  root="$(cd "${candidate}/.." 2>/dev/null && pwd -P || true)"
  [ -n "${root}" ] || return 1
  [ -d "${root}/annotations" ] || return 1

  repair_coco_symlink "${root}" || return 1

  [ -f "${candidate}/annotations/captions_val2014.json" ] || return 1
  [ -f "${candidate}/annotations/instances_val2014.json" ] || return 1
  find "${candidate}" -maxdepth 1 -type f -name 'COCO_val2014_*.jpg' -print -quit | grep -q .
}

print_coco_diagnostics() {
  local root="$1"
  local val_dir="${root}/val2014"

  echo
  echo "COCO diagnostics:"
  echo "  COCO_ROOT=${root}"
  echo "  DATA_PATH=${val_dir}"
  echo
  echo "Top-level:"
  ls -lah "${root}" || true
  echo
  echo "val2014:"
  ls -lah "${val_dir}" 2>/dev/null | head -30 || true
  echo
  echo "annotations:"
  ls -lah "${root}/annotations" 2>/dev/null | head -30 || true
  echo
  echo "val2014/annotations link:"
  ls -lah "${val_dir}/annotations" 2>/dev/null || true
  echo
  echo "Required JSON files:"
  ls -lah "${val_dir}/annotations/captions_val2014.json" 2>/dev/null || true
  ls -lah "${val_dir}/annotations/instances_val2014.json" 2>/dev/null || true
  echo
  echo "Sample image:"
  find "${val_dir}" -maxdepth 1 -type f -name 'COCO_val2014_*.jpg' -print -quit 2>/dev/null || true
  echo
}

validate_or_extract_coco() {
  local root="$1"

  echo "Validating COCO under: ${root}"

  # Existing extracted folders may be partial from a previous failed run.
  # Validate them first; if invalid, do NOT skip extraction just because dirs exist.
  if [ -d "${root}/val2014" ] && [ -d "${root}/annotations" ]; then
    repair_coco_symlink "${root}" || true
    if coco_ready_at "${root}/val2014"; then
      echo "COCO2014 ready from existing extracted folders:"
      echo "  ${root}/val2014"
      find "${root}/val2014" -maxdepth 1 -type f -name 'COCO_val2014_*.jpg' -print -quit
      return 0
    fi
    warn "Existing COCO folders are incomplete; will re-extract from validated zip files."
  fi

  ensure_zip_downloaded_and_valid "http://images.cocodataset.org/zips/val2014.zip" "${root}/val2014.zip"
  ensure_zip_downloaded_and_valid "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" "${root}/annotations_trainval2014.zip"

  extract_zip_fresh "${root}/val2014.zip" "${root}" "val2014"
  extract_zip_fresh "${root}/annotations_trainval2014.zip" "${root}" "annotations"

  repair_coco_symlink "${root}" || true

  if coco_ready_at "${root}/val2014"; then
    echo "COCO2014 ready after extraction:"
    echo "  ${root}/val2014"
    find "${root}/val2014" -maxdepth 1 -type f -name 'COCO_val2014_*.jpg' -print -quit
    return 0
  fi

  print_coco_diagnostics "${root}"
  return 1
}

find_existing_coco() {
  if coco_ready_at "${DATA_PATH}"; then
    COCO_ROOT="$(cd "${DATA_PATH}/.." && pwd -P)"
    DATA_PATH="${COCO_ROOT}/val2014"
    return 0
  fi

  if [ -d "${DATASETS_DIR}" ]; then
    while IFS= read -r cap_file; do
      ann_dir="$(dirname "${cap_file}")"
      root_dir="$(dirname "${ann_dir}")"
      val_dir="${root_dir}/val2014"

      if [ -d "${val_dir}" ] && coco_ready_at "${val_dir}"; then
        DATA_PATH="${val_dir}"
        COCO_ROOT="${root_dir}"
        return 0
      fi
    done < <(find "${DATASETS_DIR}" -path '*/annotations/captions_val2014.json' 2>/dev/null | head -50)
  fi

  return 1
}

setup_coco2014() {
  if [ "${INSTALL_COCO}" != "1" ]; then
    log "Skipping COCO2014 because INSTALL_COCO=${INSTALL_COCO}"
    return 0
  fi

  log "Checking COCO2014 data"

  mkdir -p "${COCO_ROOT}"

  if find_existing_coco; then
    echo "COCO2014 ready:"
    echo "  ${DATA_PATH}"
    find "${DATA_PATH}" -maxdepth 1 -type f -name 'COCO_val2014_*.jpg' -print -quit
    return 0
  fi

  echo "COCO2014 not ready. Downloading/validating/extracting into ${COCO_ROOT}"

  if ! validate_or_extract_coco "${COCO_ROOT}"; then
    die "COCO2014 setup failed: ${COCO_ROOT}/val2014"
  fi

  DATA_PATH="${COCO_ROOT}/val2014"
  echo "COCO2014 final DATA_PATH=${DATA_PATH}"
}

setup_weights() {
  if [ "${INSTALL_WEIGHTS}" != "1" ]; then
    log "Skipping optional weights because INSTALL_WEIGHTS=${INSTALL_WEIGHTS}"
    return 0
  fi

  log "Checking optional weights"

  mkdir -p "${WEIGHTS_DIR}"

  if find "${WEIGHTS_DIR}" -type f | grep -q .; then
    echo "Weights already exist:"
    echo "  ${WEIGHTS_DIR}"
    find "${WEIGHTS_DIR}" -maxdepth 3 -type f | head -30
    return 0
  fi

  echo "Weights missing. Trying gdown:"
  echo "  ${WEIGHTS_URL}"

  run_logged "pip_gdown.log" python -m pip install -U gdown

  set +e
  gdown --folder "${WEIGHTS_URL}" -O "${WEIGHTS_DIR}" --remaining-ok 2>&1 | tee "${LOG_DIR}/gdown_weights.log"
  local status=${PIPESTATUS[0]}
  set -e

  if [ "${status}" -ne 0 ]; then
    warn "Google Drive download failed/blocked. Continue without weights."
    warn "Manually place weights under: ${WEIGHTS_DIR}"
    return 0
  fi

  find "${WEIGHTS_DIR}" -maxdepth 3 -type f | head -50 || true
}

install_patternlite_optional() {
  if [ "${INSTALL_PATTERNLITE}" != "1" ]; then
    log "Skipping PatternLite/OpenJDK because INSTALL_PATTERNLITE=${INSTALL_PATTERNLITE}"
    return 0
  fi

  log "Installing optional OpenJDK 11 and PatternLite inside ${ENV_NAME}"

  if run_logged_allow_fail "conda_openjdk.log" conda install -c conda-forge openjdk=11 -y; then
    which java || true
    java -version || true
  else
    warn "OpenJDK install failed. Continuing."
  fi

  # Do not install full clips/pattern by default: it pulls mysqlclient and often
  # fails without system MariaDB/MySQL headers. PatternLite provides pattern.en.
  if [ "${INSTALL_PATTERN_GIT}" = "1" ]; then
    run_logged_allow_fail "pip_pattern_git.log" python -m pip install "git+https://github.com/clips/pattern.git" || \
      warn "pattern git install failed. Continuing."
  fi

  python -m pip uninstall -y pattern Pattern pattern3 PatternLite >/dev/null 2>&1 || true

  if ! run_logged_allow_fail "pip_patternlite.log" python -m pip install PatternLite; then
    warn "PatternLite install failed. Continuing."
  fi

  python - <<'PY' || true
try:
    from pattern.en import singularize
    print("PatternLite pattern.en OK:", singularize("dogs"))
except Exception as e:
    print("PatternLite pattern.en WARN:", repr(e))
PY
}

build_groundingdino() {
  log "Building GroundingDINO"

  [ -d "${GROUNDINGDINO_DIR}" ] || die "GroundingDINO dir missing: ${GROUNDINGDINO_DIR}"

  cd "${GROUNDINGDINO_DIR}"

  write_activation_hook

  which nvcc
  nvcc --version
  nvcc --version | grep -q "release ${TARGET_CUDA}" || die "nvcc mismatch before GroundingDINO build"

  rm -rf build *.egg-info groundingdino.egg-info

  set +e
  MAX_JOBS=1 python -m pip install -e . --no-build-isolation --no-cache-dir -v 2>&1 | tee "${LOG_DIR}/build_groundingdino.log"
  local status=${PIPESTATUS[0]}
  set -e

  if [ "${status}" -ne 0 ]; then
    echo
    echo "GroundingDINO build failed. Key error lines:"
    grep -n -i -E "fatal error|error:|FAILED:|unsupported|killed|No such|undefined|nvcc|sm_|gcc|g\+\+|ld:|mismatch" "${LOG_DIR}/build_groundingdino.log" | head -240 || true
    die "GroundingDINO build failed: ${LOG_DIR}/build_groundingdino.log"
  fi
}

verify_final() {
  log "Final verification"

  verify_torch_cuda
  verify_runtime_imports

  cd "${RVCD_MAIN_CODES}"
  export PYTHONPATH="${RVCD_MAIN_CODES}:${PYTHONPATH:-}"

  python - <<'PY'
from groundingdino.util.inference import load_model
print("GroundingDINO load_model import OK")
PY

  echo
  echo "DONE"
  echo "Mode: ${MODE}"
  echo "Env: ${ENV_NAME}"
  echo "GPU: ${GPU_NAME}"
  echo "CUDA: ${TARGET_CUDA}"
  echo "Torch: ${TORCH_VERSION}"
  echo "TorchVision: ${TORCHVISION_VERSION}"
  echo "TorchAudio: ${TORCHAUDIO_VERSION}"
  echo "TORCH_CUDA_ARCH_LIST: ${TARGET_ARCH_LIST}"
  echo "BNB_PACKAGE: ${BNB_PACKAGE}"
  echo "BNB_CUDA_VERSION/CUDA_VERSION: ${BNB_CUDA_CODE}"
  echo "COCO data path: ${DATA_PATH}"
  echo "Weights dir: ${WEIGHTS_DIR}"
  echo "Logs: ${LOG_DIR}"
  echo
  echo "Run:"
  echo "  conda activate ${ENV_NAME}"
  echo "  cd ${RVCD_MAIN_CODES}"
  echo "  bash run_example.sh"
  echo
  if [ "${STRICT_BNB}" != "1" ]; then
    echo "Note:"
    echo "  bitsandbytes is non-fatal in this script. If you need quantization paths,"
    echo "  test with:"
    echo "    conda activate ${ENV_NAME}"
    echo "    BNB_CUDA_VERSION=${BNB_CUDA_CODE} CUDA_VERSION=${BNB_CUDA_CODE} python -c 'import bitsandbytes; print(\"bnb ok\")'"
  fi
}

main() {
  [ -d "${RVCD_ROOT}" ] || die "RVCD_ROOT not found: ${RVCD_ROOT}"
  [ -d "${RVCD_MAIN_CODES}" ] || die "RVCD_MAIN_CODES not found: ${RVCD_MAIN_CODES}"

  init_logs
  load_conda
  detect_gpu0
  create_clean_env
  install_torch_stack

  # Critical: install/verify runtime deps before CUDA extension build.
  install_runtime_deps_first

  # Optional Java/PatternLite, data, and weights.
  install_patternlite_optional
  setup_coco2014
  setup_weights

  # CUDA build tools and GroundingDINO are last.
  install_cuda_build_tools_after_deps
  build_groundingdino
  verify_final
}

main "$@"
