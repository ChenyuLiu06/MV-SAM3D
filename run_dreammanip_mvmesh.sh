#!/usr/bin/env bash
set -euo pipefail

INPUT_ROOT=/home/yilin/chenyu/dreammanip2/mvmesh
MVSAM_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MVSAM_ENV=/mnt/conda/yilin/sam3d_5090
SAM2_RUNNER=/home/yilin/chenyu/sam-3d-objects-5090/run_dreammanip_multiview_segment.sh
CHECKPOINTS_ROOT=/home/yilin/chenyu/sam-3d-objects-5090/checkpoints
RUNTIME_ROOT="$MVSAM_REPO/data/dreammanip_runtime"

usage() {
  cat <<EOF
usage: $0 GPU_ID [--instance NAME]

examples:
  $0 0
  $0 0 --instance fuelcan

The script runs, in order:
  1. SAM2 masks from daid_config.json point annotations
  2. mask-only metric pointmaps (background points set to NaN)
  3. native MV-SAM3D Stage 1: RGB + mask + mask-only pointmaps -> sparse structure
  4. native MV-SAM3D Stage 2: structure + RGB + mask -> mesh/Gaussian
  5. predicted metric pose applied and final mesh written to INSTANCE/mesh.glb
EOF
}

if (( $# == 0 )); then
  usage >&2
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi
GPU_ID=$1
shift
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be a non-negative integer" >&2; exit 2; }

INSTANCE=
while (( $# )); do
  case "$1" in
    --instance)
      (( $# >= 2 )) || { echo "--instance requires a name" >&2; exit 2; }
      INSTANCE=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -d "$INPUT_ROOT" ]] || { echo "input root not found: $INPUT_ROOT" >&2; exit 1; }
[[ -d "$MVSAM_REPO" ]] || { echo "MV-SAM3D repo not found: $MVSAM_REPO" >&2; exit 1; }
[[ -x "$MVSAM_ENV/bin/python" ]] || { echo "Python environment not found: $MVSAM_ENV" >&2; exit 1; }
[[ -x "$SAM2_RUNNER" ]] || { echo "SAM2 runner not found: $SAM2_RUNNER" >&2; exit 1; }
[[ -f "$CHECKPOINTS_ROOT/hf-5090/pipeline.yaml" ]] || {
  echo "MV-SAM3D checkpoints not found: $CHECKPOINTS_ROOT/hf-5090/pipeline.yaml" >&2
  exit 1
}

declare -a INSTANCE_DIRS=()
if [[ -n "$INSTANCE" ]]; then
  [[ "$INSTANCE" != */* && "$INSTANCE" != "." && "$INSTANCE" != ".." ]] || {
    echo "invalid instance name: $INSTANCE" >&2
    exit 2
  }
  INSTANCE_DIR="$INPUT_ROOT/$INSTANCE"
  [[ -d "$INSTANCE_DIR" ]] || { echo "instance not found: $INSTANCE_DIR" >&2; exit 1; }
  INSTANCE_DIRS+=("$INSTANCE_DIR")
else
  while IFS= read -r -d '' INSTANCE_DIR; do
    if find "$INSTANCE_DIR" -mindepth 1 -maxdepth 1 -type d -name 'view_*' -print -quit | grep -q .; then
      INSTANCE_DIRS+=("$INSTANCE_DIR")
    fi
  done < <(find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

(( ${#INSTANCE_DIRS[@]} > 0 )) || { echo "no instances found under: $INPUT_ROOT" >&2; exit 1; }

NEED_SAM2=false
for INSTANCE_DIR in "${INSTANCE_DIRS[@]}"; do
  VIEW_COUNT=0
  while IFS= read -r -d '' VIEW_DIR; do
    ((VIEW_COUNT += 1))
    for REQUIRED in rgb.png depth_aligned_m.npy intrinsic.yaml; do
      [[ -s "$VIEW_DIR/$REQUIRED" ]] || { echo "missing: $VIEW_DIR/$REQUIRED" >&2; exit 1; }
    done
    if [[ ! -s "$VIEW_DIR/sam2_mask.png" ]]; then
      [[ -s "$VIEW_DIR/daid_config.json" ]] || {
        echo "mask and point annotation are both missing: $VIEW_DIR" >&2
        exit 1
      }
      NEED_SAM2=true
    fi
  done < <(find "$INSTANCE_DIR" -mindepth 1 -maxdepth 1 -type d -name 'view_*' -print0 | sort -z)
  (( VIEW_COUNT >= 2 )) || { echo "fewer than two views: $INSTANCE_DIR" >&2; exit 1; }
done

if [[ "$NEED_SAM2" == true ]]; then
  if [[ -n "$INSTANCE" ]]; then
    "$SAM2_RUNNER" "$INPUT_ROOT" "$GPU_ID" --sample "$INSTANCE"
  else
    "$SAM2_RUNNER" "$INPUT_ROOT" "$GPU_ID"
  fi
else
  echo "all selected views already have SAM2 masks; skipping SAM2"
fi

export CONDA_PREFIX="$MVSAM_ENV"
export CUDA_HOME="$MVSAM_ENV"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LIDRA_SKIP_INIT=true
export XFORMERS_DISABLED=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="$MVSAM_REPO:$MVSAM_REPO/notebook"
export MPLCONFIGDIR=/tmp/yilin-mpl
export TORCH_EXTENSIONS_DIR=/mnt/conda/yilin/torch-extensions
export LD_LIBRARY_PATH="$MVSAM_ENV/targets/x86_64-linux/lib:$MVSAM_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! -e "$MVSAM_REPO/checkpoints" && ! -L "$MVSAM_REPO/checkpoints" ]]; then
  ln -s "$CHECKPOINTS_ROOT" "$MVSAM_REPO/checkpoints"
fi
[[ -f "$MVSAM_REPO/checkpoints/hf-5090/pipeline.yaml" ]] || {
  echo "MV-SAM3D checkpoint link is invalid: $MVSAM_REPO/checkpoints" >&2
  exit 1
}

mkdir -p "$RUNTIME_ROOT"
cd "$MVSAM_REPO"

for INSTANCE_DIR in "${INSTANCE_DIRS[@]}"; do
  INSTANCE_NAME=$(basename "$INSTANCE_DIR")
  RUNTIME_DIR="$RUNTIME_ROOT/$INSTANCE_NAME"

  echo "[$INSTANCE_NAME] building masks + mask-only metric pointmaps input"
  "$MVSAM_ENV/bin/python" - "$INSTANCE_DIR" "$RUNTIME_DIR" <<'PY'
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from preprocessing.dreammanip_adapter import load_dreammanip_sample

instance_dir = Path(sys.argv[1]).resolve()
runtime_dir = Path(sys.argv[2]).resolve()
images_dir = runtime_dir / "images"
masks_dir = runtime_dir / "object"
images_dir.mkdir(parents=True, exist_ok=True)
masks_dir.mkdir(parents=True, exist_ok=True)

for directory in (images_dir, masks_dir):
    for old_file in directory.glob("*.png"):
        old_file.unlink()

sample = load_dreammanip_sample(instance_dir, camera_pose_config=None)
pointmaps = []
intrinsics = []
image_files = []

for index, view in enumerate(sample.views):
    image_name = str(index)
    image_path = images_dir / f"{image_name}.png"
    mask_path = masks_dir / f"{image_name}.png"

    Image.fromarray(view.image).save(image_path)
    alpha = (view.mask.astype(np.uint8) * 255)[..., None]
    rgba = np.concatenate((view.image, alpha), axis=2)
    Image.fromarray(rgba, mode="RGBA").save(mask_path)

    pointmap = view.pointmap.copy()
    pointmap[:, ~view.mask] = np.nan

    np.save(view.directory / "pointmap.npy", pointmap)
    pointmaps.append(pointmap)
    intrinsics.append(view.intrinsics)
    image_files.append(image_path.name)

np.savez_compressed(
    runtime_dir / "real_pointmaps.npz",
    pointmaps_sam3d=np.stack(pointmaps, axis=0),
    intrinsics=np.stack(intrinsics, axis=0),
    image_files=np.asarray(image_files),
)
print(f"prepared {len(pointmaps)} views in {runtime_dir}")
PY

  for VIEW_DIR in "$INSTANCE_DIR"/view_*; do
    [[ -s "$VIEW_DIR/sam2_mask.png" ]] || { echo "missing SAM2 mask: $VIEW_DIR" >&2; exit 1; }
    [[ -s "$VIEW_DIR/pointmap.npy" ]] || { echo "missing pointmap: $VIEW_DIR" >&2; exit 1; }
  done
  [[ -s "$RUNTIME_DIR/real_pointmaps.npz" ]] || { echo "pointmap package was not created" >&2; exit 1; }

  echo "[$INSTANCE_NAME] Stage 1 + Stage 2: running native MV-SAM3D on GPU $GPU_ID"
  echo "[$INSTANCE_NAME]   Stage 1 input: RGB + SAM2 mask + mask-only pointmap"
  echo "[$INSTANCE_NAME]   Stage 2 input: Stage 1 structure + RGB + SAM2 mask"
  "$MVSAM_ENV/bin/python" run_inference_weighted.py \
    --input_path "$RUNTIME_DIR" \
    --mask_prompt object \
    --model_tag hf-5090 \
    --decode_formats gaussian,mesh \
    --seed 42 \
    --stage1_steps 50 \
    --stage2_steps 25 \
    --da3_output "$RUNTIME_DIR/real_pointmaps.npz" \
    --no_stage1_weighting \
    --no_stage2_weighting

  VISUALIZATION_ROOT="$MVSAM_REPO/visualization/$INSTANCE_NAME/object"
  RESULT_GLB=$(find "$VISUALIZATION_ROOT" -type f -name result.glb -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
  [[ -n "$RESULT_GLB" && -s "$RESULT_GLB" ]] || { echo "result.glb not found for $INSTANCE_NAME" >&2; exit 1; }

  PARAMS_NPZ="$(dirname "$RESULT_GLB")/params.npz"
  [[ -s "$PARAMS_NPZ" ]] || { echo "pose parameters not found: $PARAMS_NPZ" >&2; exit 1; }

  echo "[$INSTANCE_NAME] applying predicted metric pose to mesh"
  "$MVSAM_ENV/bin/python" - "$RESULT_GLB" "$PARAMS_NPZ" "$INSTANCE_DIR/mesh.glb" <<'PY'
import sys
from pathlib import Path

import numpy as np
import trimesh

source_glb = Path(sys.argv[1]).resolve()
params_path = Path(sys.argv[2]).resolve()
output_glb = Path(sys.argv[3]).resolve()

params = np.load(params_path, allow_pickle=False)
required = ("scale", "rotation", "translation")
missing = [name for name in required if name not in params]
if missing:
    raise KeyError(f"{params_path}: missing {', '.join(missing)}")

scale = np.asarray(params["scale"], dtype=np.float64).reshape(-1)
rotation = np.asarray(params["rotation"], dtype=np.float64).reshape(-1)
translation = np.asarray(params["translation"], dtype=np.float64).reshape(-1)
if scale.size == 1:
    scale = np.repeat(scale, 3)
if scale.size < 3 or rotation.size < 4 or translation.size < 3:
    raise ValueError(
        f"invalid pose shapes: scale={scale.shape}, rotation={rotation.shape}, "
        f"translation={translation.shape}"
    )
scale = scale[:3]
rotation = rotation[:4]  # SAM3D uses quaternion order [w, x, y, z]
translation = translation[:3]

norm = np.linalg.norm(rotation)
if not np.isfinite(norm) or norm < 1e-12:
    raise ValueError(f"invalid rotation quaternion: {rotation}")
w, x, y, z = rotation / norm
rotation_matrix = np.array(
    [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ],
    dtype=np.float64,
)

scene = trimesh.load(str(source_glb), force="scene")
geometry_count = 0
vertex_count = 0
for geometry in scene.geometry.values():
    if not hasattr(geometry, "vertices"):
        continue
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    # result.glb is already in the mesh's Y-up canonical orientation. The
    # predicted pose maps it directly into the first-view metric camera frame.
    geometry.vertices = (vertices * scale) @ rotation_matrix.T + translation
    geometry_count += 1
    vertex_count += len(vertices)

if geometry_count == 0 or vertex_count == 0:
    raise ValueError(f"no mesh geometry found in {source_glb}")

scene.export(str(output_glb))
print(
    f"wrote {output_glb} from {source_glb}; geometries={geometry_count}, "
    f"vertices={vertex_count}, scale={scale.tolist()}, translation={translation.tolist()}"
)
PY

  [[ -s "$INSTANCE_DIR/mesh.glb" ]] || { echo "metric mesh was not created: $INSTANCE_DIR/mesh.glb" >&2; exit 1; }
  echo "[$INSTANCE_NAME] final mesh: $INSTANCE_DIR/mesh.glb"
done

echo "completed ${#INSTANCE_DIRS[@]} instance(s)"
