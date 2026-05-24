#!/usr/bin/env bash
# setup_rvcd_reliable_all_in_one.sh
#
# Reliable RVCD all-in-one setup.
#
# Fixed design:
#   1. Detect GPU 0.
#   2. Always delete/recreate the target env.
#   3. Install GPU-compatible torch.
#   4. Install and verify RVCD runtime pip dependencies FIRST.
#      This means tqdm/transformers/ultralytics/minigpt4 deps are installed
#      before any GroundingDINO/CUDA-extension build step can fail.
#   5. Only after runtime deps are verified, install CUDA build tools.
#   6. Build GroundingDINO.
#   7. Final verification.
#
# If anything fails, logs are written under:
#   /home/jihoon/jihoon/RVCD/setup_logs
#
# Usage:
#   cd /home/jihoon/jihoon/RVCD
#   bash setup_rvcd_reliable_all_in_one.sh
#
# Optional:
#   INSTALL_COCO=0 bash setup_rvcd_reliable_all_in_one.sh
#   INSTALL_WEIGHTS=0 bash setup_rvcd_reliable_all_in_one.sh

set -Eeo pipefail

# ----------------------------
# Paths
# ----------------------------
RVCD_ROOT="${RVCD_ROOT:-/home/jihoon/jihoon/RVCD}"
RVCD_MAIN_CODES="${RVCD_MAIN_CODES:-${RVCD_ROOT}/MAIN_CODES}"
GROUNDINGDINO_DIR="${GROUNDINGDINO_DIR:-${RVCD_MAIN_CODES}/decoder_zoo/GroundingDINO}"

DATASETS_DIR="${DATASETS_DIR:-/home/jihoon/jihoon/DATASETS}"
COCO_ROOT="${COCO_ROOT:-${DATASETS_DIR}/coco2014}"
DATA_PATH="${DATA_PATH:-${COCO_ROOT}/val2014}"

TRANSFORMERS_DIR="${TRANSFORMERS_DIR:-${RVCD_ROOT}/transformers-4.36.2}"
TRANSFORMERS_ALT_DIR="${TRANSFORMERS_ALT_DIR:-${RVCD_MAIN_CODES}/transformers-4.36.2}"

INSTALL_COCO="${INSTALL_COCO:-1}"
INSTALL_WEIGHTS="${INSTALL_WEIGHTS:-1}"
WEIGHTS_URL="${WEIGHTS_URL:-https://drive.google.com/drive/folders/1UaMJga-BKju88CXAdonbiQujBKkdcVGX}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${RVCD_MAIN_CODES}/decoder_zoo/weights}"

LOG_DIR="${LOG_DIR:-${RVCD_ROOT}/setup_logs}"

# ----------------------------
# GPU-dependent stacks
# ----------------------------
BW_ENV="${BW_ENV:-RVCD_BW}"
BW_PYTHON="${BW_PYTHON:-3.10}"
BW_TORCH="${BW_TORCH:-2.7.1}"
BW_TORCHVISION="${BW_TORCHVISION:-0.22.1}"
BW_TORCHAUDIO="${BW_TORCHAUDIO:-2.7.1}"
BW_CUDA="${BW_CUDA:-12.8}"
BW_ARCH_LIST="${BW_ARCH_LIST:-12.0}"

BASE_ENV="${BASE_ENV:-RVCD}"
BASE_PYTHON="${BASE_PYTHON:-3.9}"
BASE_TORCH="${BASE_TORCH:-2.0.0}"
BASE_TORCHVISION="${BASE_TORCHVISION:-0.15.1}"
BASE_TORCHAUDIO="${BASE_TORCHAUDIO:-2.0.1}"
BASE_CUDA="${BASE_CUDA:-11.7}"

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
    grep -n -i -E "error|failed|conflict|resolution|No matching|Could not|subprocess|metadata|wheel|traceback|exception" "${LOG_DIR}/${logfile}" | tail -80 || true
    exit "${status}"
  fi
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
    BNB_PACKAGE="bitsandbytes>=0.46.0"
  else
    MODE="base"
    ENV_NAME="${BASE_ENV}"
    PYTHON_VERSION="${BASE_PYTHON}"
    TARGET_CUDA="${BASE_CUDA}"
    TORCH_VERSION="${BASE_TORCH}"
    TORCHVISION_VERSION="${BASE_TORCHVISION}"
    TORCHAUDIO_VERSION="${BASE_TORCHAUDIO}"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu117"
    TARGET_ARCH_LIST="8.6"
    BNB_PACKAGE="bitsandbytes==0.37.0"

    if [ -n "${GPU_CC:-}" ] && [[ "${GPU_CC}" =~ ^[0-9]+\.[0-9]+$ ]]; then
      major="${GPU_CC%%.*}"
      minor="${GPU_CC#*.}"
      if [ "${major}" -lt 8 ] || { [ "${major}" -eq 8 ] && [ "${minor}" -le 6 ]; }; then
        TARGET_ARCH_LIST="${GPU_CC}"
      else
        TARGET_ARCH_LIST="8.6+PTX"
      fi
    fi
  fi

  cat <<EOF
Selected setup:
  MODE=${MODE}
  ENV_NAME=${ENV_NAME}
  PYTHON_VERSION=${PYTHON_VERSION}
  TORCH_VERSION=${TORCH_VERSION}
  TORCH_INDEX_URL=${TORCH_INDEX_URL}
  TARGET_CUDA=${TARGET_CUDA}
  TORCH_CUDA_ARCH_LIST=${TARGET_ARCH_LIST}
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

install_runtime_deps_first() {
  log "Installing RVCD runtime dependencies BEFORE CUDA/GroundingDINO build"

  # This requirements file is saved under the repo, not /tmp, so it remains debuggable.
  REQ_FILE="${LOG_DIR}/rvcd_full_deps_no_torch_runtime.txt"

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
${BNB_PACKAGE}
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

  # Install all runtime dependencies. If this fails, stop here before CUDA build.
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
  write_activation_hook
  verify_runtime_imports
}

write_activation_hook() {
  log "Writing persistent PYTHONPATH and build env"

  mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"

  cat > "${CONDA_PREFIX}/etc/conda/activate.d/rvcd_env.sh" <<EOF
export CUDA_HOME=\$CONDA_PREFIX
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CONDA_PREFIX/targets/x86_64-linux/lib:\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=\$CONDA_PREFIX/targets/x86_64-linux/lib:\$CONDA_PREFIX/lib:\${LIBRARY_PATH:-}
export CC=\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export TORCH_CUDA_ARCH_LIST="${TARGET_ARCH_LIST}"
export PYTHONPATH=${RVCD_MAIN_CODES}:\${PYTHONPATH:-}
EOF

  export CUDA_HOME="${CONDA_PREFIX}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
  export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
  export TORCH_CUDA_ARCH_LIST="${TARGET_ARCH_LIST}"
  export PYTHONPATH="${RVCD_MAIN_CODES}:${PYTHONPATH:-}"

  echo "PYTHONPATH=${PYTHONPATH}"
}

verify_runtime_imports() {
  log "Verifying runtime imports BEFORE CUDA build"

  cd "${RVCD_MAIN_CODES}"
  export PYTHONPATH="${RVCD_MAIN_CODES}:${PYTHONPATH:-}"

  python - <<PY
import importlib
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
    ("bitsandbytes", "bitsandbytes"),
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
      -c nvidia/label/cuda-11.7.0 \
      -c nvidia \
      "cuda-nvcc>=11.7,<11.8" \
      "cuda-cudart>=11.7,<11.8" \
      "cuda-cudart-dev>=11.7,<11.8" \
      "cuda-libraries-dev>=11.7,<11.8" \
      "cuda-thrust>=11.7,<11.8" \
      "libcusparse-dev>=11.7,<11.8" \
      "libcublas-dev>=11.10,<12" \
      "libcusolver-dev>=11.4,<11.5"
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

  if [ -f "$out" ]; then
    echo "Already exists: $out"
    return 0
  fi

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

coco_ready_at() {
  local candidate="$1"
  [ -d "${candidate}" ] || return 1
  [ -f "${candidate}/annotations/captions_val2014.json" ] || return 1
  [ -f "${candidate}/annotations/instances_val2014.json" ] || return 1
  find "${candidate}" -maxdepth 1 -name 'COCO_val2014_*.jpg' | grep -q .
}

find_existing_coco() {
  if coco_ready_at "${DATA_PATH}"; then
    return 0
  fi

  if [ -d "${DATASETS_DIR}" ]; then
    while IFS= read -r cap_file; do
      ann_dir="$(dirname "${cap_file}")"
      root_dir="$(dirname "${ann_dir}")"
      val_dir="${root_dir}/val2014"

      if [ -d "${val_dir}" ] && find "${val_dir}" -maxdepth 1 -name 'COCO_val2014_*.jpg' | grep -q .; then
        ln -sfn ../annotations "${val_dir}/annotations"
        if coco_ready_at "${val_dir}"; then
          DATA_PATH="${val_dir}"
          COCO_ROOT="${root_dir}"
          return 0
        fi
      fi
    done < <(find "${DATASETS_DIR}" -path '*/annotations/captions_val2014.json' 2>/dev/null | head -20)
  fi

  return 1
}

setup_coco2014() {
  if [ "${INSTALL_COCO}" != "1" ]; then
    log "Skipping COCO2014 because INSTALL_COCO=${INSTALL_COCO}"
    return 0
  fi

  log "Checking COCO2014 data"

  if find_existing_coco; then
    echo "COCO2014 ready:"
    echo "  ${DATA_PATH}"
    find "${DATA_PATH}" -maxdepth 1 -name 'COCO_val2014_*.jpg' | head -1
    return 0
  fi

  echo "COCO2014 missing. Downloading into ${COCO_ROOT}"
  mkdir -p "${COCO_ROOT}"
  cd "${COCO_ROOT}"

  download_file "http://images.cocodataset.org/zips/val2014.zip" "val2014.zip"
  download_file "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" "annotations_trainval2014.zip"

  [ -d "${COCO_ROOT}/val2014" ] || python -m zipfile -e "${COCO_ROOT}/val2014.zip" "${COCO_ROOT}"
  [ -d "${COCO_ROOT}/annotations" ] || python -m zipfile -e "${COCO_ROOT}/annotations_trainval2014.zip" "${COCO_ROOT}"

  ln -sfn ../annotations "${COCO_ROOT}/val2014/annotations"
  DATA_PATH="${COCO_ROOT}/val2014"

  coco_ready_at "${DATA_PATH}" || die "COCO2014 setup failed: ${DATA_PATH}"
  echo "COCO2014 ready: ${DATA_PATH}"
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

build_groundingdino() {
  log "Building GroundingDINO"

  [ -d "${GROUNDINGDINO_DIR}" ] || die "GroundingDINO dir missing: ${GROUNDINGDINO_DIR}"

  cd "${GROUNDINGDINO_DIR}"

  export CUDA_HOME="${CONDA_PREFIX}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
  export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
  export TORCH_CUDA_ARCH_LIST="${TARGET_ARCH_LIST}"
  export PYTHONPATH="${RVCD_MAIN_CODES}:${PYTHONPATH:-}"

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
  echo "COCO data path: ${DATA_PATH}"
  echo "Weights dir: ${WEIGHTS_DIR}"
  echo "Logs: ${LOG_DIR}"
  echo
  echo "Run:"
  echo "  conda activate ${ENV_NAME}"
  echo "  cd ${RVCD_MAIN_CODES}"
  echo "  bash run_example.sh"
}

main() {
  [ -d "${RVCD_ROOT}" ] || die "RVCD_ROOT not found: ${RVCD_ROOT}"
  [ -d "${RVCD_MAIN_CODES}" ] || die "RVCD_MAIN_CODES not found: ${RVCD_MAIN_CODES}"

  init_logs
  load_conda
  detect_gpu0
  create_clean_env
  install_torch_stack

  # Critical fix: install/verify runtime deps before CUDA extension build.
  install_runtime_deps_first

  # Optional data/weights are safe after runtime deps.
  setup_coco2014
  setup_weights

  # CUDA build tools and GroundingDINO are last.
  install_cuda_build_tools_after_deps
  build_groundingdino
  verify_final
}

main "$@"


conda activate RVCD_BW

conda install -c conda-forge openjdk=11 -y

which java
java -version

python -m pip install "git+https://github.com/clips/pattern.git"


# 잘못 깔린 pattern 제거
python -m pip uninstall -y pattern Pattern pattern3 PatternLite

# pattern.en 제공하는 패키지 설치
python -m pip install PatternLite

