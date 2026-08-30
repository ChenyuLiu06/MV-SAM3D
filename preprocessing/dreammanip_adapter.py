"""DreamManip capture adapter for MV-SAM3D.

This module only prepares data. It does not run SAM2, DA3, or MV-SAM3D.

The adapter keeps the capture data in the camera convention used by the
calibration files:

    pointmap shape: (3, H, W)
    pointmap axes: X right, Y down, Z forward
    pointmap unit: meter

MV-SAM3D can convert this external pointmap to its internal convention at the
model boundary. Camera poses are deliberately loaded but not applied yet: the
calibrated extrinsics and the camera-to-object convention will be filled in
after the calibration format is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml
from PIL import Image


DEFAULT_INPUT_ROOT = Path("/home/yilin/chenyu/dreammanip2/mvmesh")
DEFAULT_CAMERA_POSE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "dreammanip_camera_poses.yaml"
)

REQUIRED_VIEW_FILES = (
    "rgb.png",
    "sam2_mask.png",
    "depth_aligned_m.npy",
    "intrinsic.yaml",
)


@dataclass(frozen=True)
class DreamManipView:
    """Prepared input for one capture view."""

    name: str
    directory: Path
    image: np.ndarray
    mask: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    pointmap: np.ndarray
    camera_pose: Optional[np.ndarray]


@dataclass(frozen=True)
class DreamManipSample:
    """All prepared views belonging to one DreamManip sample."""

    name: str
    directory: Path
    views: Sequence[DreamManipView]
    camera_pose_config: Optional[Path]

    @property
    def view_images(self) -> List[np.ndarray]:
        return [view.image for view in self.views]

    @property
    def view_masks(self) -> List[np.ndarray]:
        return [view.mask for view in self.views]

    @property
    def view_pointmaps(self) -> List[np.ndarray]:
        return [view.pointmap for view in self.views]

    @property
    def view_camera_poses(self) -> List[Optional[np.ndarray]]:
        return [view.camera_pose for view in self.views]

    def as_mv_sam3d_inputs(self) -> Dict[str, List[object]]:
        """Return the four lists expected by the future MV-SAM3D runner.

        ``view_camera_poses`` is intentionally not passed to the current
        ``run_multi_view`` call yet. It is exposed here so the later pose
        conversion can be added without changing the data reader.
        """

        return {
            "view_images": self.view_images,
            "view_masks": self.view_masks,
            "view_pointmaps": self.view_pointmaps,
            "view_camera_poses": self.view_camera_poses,
        }


def depth_to_pointmap(depth_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Unproject an aligned depth image into a metric camera pointmap."""

    depth_m = np.asarray(depth_m, dtype=np.float32)
    if depth_m.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth_m.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {intrinsics.shape}")

    height, width = depth_m.shape
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    z = depth_m
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
    pointmap = np.stack((x, y, z), axis=0)

    invalid = ~np.isfinite(z) | (z <= 0)
    pointmap[:, invalid] = np.nan
    return pointmap.astype(np.float32, copy=False)


def _load_color_intrinsics(path: Path, width: int, height: int) -> np.ndarray:
    """Read the color K because ``depth_aligned_m.npy`` is aligned to color."""

    with path.open(encoding="utf-8") as file:
        calibration = yaml.safe_load(file)

    camera_name = calibration.get("aligned_depth_intrinsics", "color")
    camera = calibration[camera_name]
    calibration_size = (camera["image_width"], camera["image_height"])
    image_size = (width, height)
    if calibration_size != image_size:
        raise ValueError(
            f"{path}: calibration size {calibration_size} does not match "
            f"image size {image_size}"
        )

    return np.asarray(camera["camera_matrix"]["data"], dtype=np.float32).reshape(3, 3)


def _load_camera_pose_table(path: Optional[Path]) -> Dict[str, Optional[np.ndarray]]:
    """Load fixed per-view poses, leaving absent entries as ``None``."""

    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    raw_views = config.get("views", {})
    poses: Dict[str, Optional[np.ndarray]] = {}
    for view_name, value in raw_views.items():
        if value is None:
            poses[view_name] = None
            continue
        matrix = value.get("matrix") if isinstance(value, dict) else value
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"{path}: pose for {view_name} must be a 4x4 matrix, "
                f"got {matrix.shape}"
            )
        poses[view_name] = matrix
    return poses


def _view_directories(sample_directory: Path) -> List[Path]:
    views = sorted(
        path for path in sample_directory.glob("view_*") if path.is_dir()
    )
    if len(views) < 2:
        raise ValueError(
            f"{sample_directory}: expected at least two view_* directories, "
            f"found {len(views)}"
        )
    return views


def load_dreammanip_sample(
    sample_directory: Path,
    camera_pose_config: Optional[Path] = DEFAULT_CAMERA_POSE_CONFIG,
    require_camera_poses: bool = False,
) -> DreamManipSample:
    """Read one existing DreamManip sample without invoking any model."""

    sample_directory = Path(sample_directory).expanduser().resolve()
    if not sample_directory.is_dir():
        raise FileNotFoundError(f"sample directory not found: {sample_directory}")

    pose_path = (
        Path(camera_pose_config).expanduser().resolve()
        if camera_pose_config is not None
        else None
    )
    pose_table = _load_camera_pose_table(pose_path)
    views: List[DreamManipView] = []

    for view_directory in _view_directories(sample_directory):
        missing = [
            filename
            for filename in REQUIRED_VIEW_FILES
            if not (view_directory / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{view_directory}: missing {', '.join(missing)}"
            )

        image = np.asarray(Image.open(view_directory / "rgb.png").convert("RGB"))
        mask = np.asarray(Image.open(view_directory / "sam2_mask.png")) > 0
        depth_m = np.asarray(np.load(view_directory / "depth_aligned_m.npy"))
        if mask.shape != image.shape[:2] or depth_m.shape != image.shape[:2]:
            raise ValueError(
                f"{view_directory}: RGB={image.shape[:2]}, mask={mask.shape}, "
                f"depth={depth_m.shape}; all must match"
            )

        intrinsics = _load_color_intrinsics(
            view_directory / "intrinsic.yaml", image.shape[1], image.shape[0]
        )
        pointmap = depth_to_pointmap(depth_m, intrinsics)
        pose = pose_table.get(view_directory.name)
        if require_camera_poses and pose is None:
            raise ValueError(
                f"camera pose for {view_directory.name} is not filled in "
                f"{pose_path}; pose application is intentionally not implemented yet"
            )

        valid_object_depth = mask & np.isfinite(depth_m) & (depth_m > 0)
        if int(valid_object_depth.sum()) == 0:
            raise ValueError(f"{view_directory}: mask has no valid depth pixels")

        views.append(
            DreamManipView(
                name=view_directory.name,
                directory=view_directory,
                image=image,
                mask=mask,
                depth_m=depth_m.astype(np.float32, copy=False),
                intrinsics=intrinsics,
                pointmap=pointmap,
                camera_pose=pose,
            )
        )

    return DreamManipSample(
        name=sample_directory.name,
        directory=sample_directory,
        views=tuple(views),
        camera_pose_config=pose_path,
    )


def camera_to_object(
    pointmap: np.ndarray,
    camera_pose: Optional[np.ndarray],
) -> np.ndarray:
    """Reserved for calibrated camera-to-object conversion.

    The external calibration convention and the object-frame definition have
    not been supplied yet. Keeping this explicit prevents silently applying an
    incorrect inverse or axis convention.
    """

    del pointmap, camera_pose
    raise NotImplementedError(
        "camera-to-object conversion will be added after the calibrated "
        "extrinsic convention is provided"
    )


__all__ = [
    "DEFAULT_CAMERA_POSE_CONFIG",
    "DEFAULT_INPUT_ROOT",
    "DreamManipSample",
    "DreamManipView",
    "camera_to_object",
    "depth_to_pointmap",
    "load_dreammanip_sample",
]
