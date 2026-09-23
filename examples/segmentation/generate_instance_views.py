import argparse
import json
import math
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import trimesh.transformations as tf

try:
    from gsplat.rendering import rasterization

    HAS_GSPLAT = True
except ImportError:
    rasterization = None
    HAS_GSPLAT = False
    print(
        "[Warning] gsplat not found. Image rendering will be skipped, but camera poses will be saved."
    )

HARDCODED_IGNORED_IDS = {
    1,#
    6,#
    7,#
    10,#
    11,#
    29,#
    33,#
    37,#
    48,#
    55,#
    58,#
    59,#
    60,#
    62,#
    64,#
    68,#
    76,#
    84,#
    85,#
    87,#
    94,#
    97,#
    99,#
    102,#
    104,#
    105,#
    106,#
    116,#
    117,#
    118,#
    123,#
    125,#
    126,#
    131,#
    132,#
    133,#
    136,#
    137,#
    138,#
    140,#
    143,#
    144,#
    146,#
}

# Hardcoded merge pairs from VLM results.
# Add more pairs as needed, e.g., (u, v).
MERGE_PAIRS: List[Tuple[int, int]] = [
    (131, 134),
    (19, 43),
    (71, 96),
    (53, 54),
    (40, 103),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render per-instance unit-sphere heatmap and generate candidate views."
    )
    parser.add_argument("--legacy_scene_rules", action="store_true",
                        help="Apply the original scene-specific merge/ignore IDs.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Scene directory containing cluster_result outputs.",
    )
    parser.add_argument(
        "--instance_id",
        type=int,
        default=None,
        help="Instance id to process. If omitted, processes all instances in atomic_geometry.pt.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "Path to the gaussian .pt checkpoint. If not provided, tries to find one in data_dir. "
            "If absent or gsplat is missing, skips rendering (still saves cameras.json)."
        ),
    )
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument(
        "--sharpness",
        type=float,
        default=8.0,
        help=(
            "Sharpness of the heatmap lobe. Higher values (e.g. 8-10) better reflect "
            "3DGS view-dependency."
        ),
    )
    parser.add_argument("--mask_id_offset", type=int, default=1)
    parser.add_argument("--use_mask_weight", action="store_true")
    parser.add_argument(
        "--top_visible",
        type=int,
        default=50,
        help="Use only the top-N views with the largest mask area for visibility.",
    )
    parser.add_argument(
        "--gen_candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable candidate view generation.",
    )
    parser.add_argument(
        "--candidate_mode",
        type=str,
        default="heatmap",
        choices=["heatmap", "geometric", "all"],
        help=(
            "View sampling strategy: 'heatmap' (visibility based), 'geometric' (OBB shape based), "
            "or 'all' (both)."
        ),
    )
    parser.add_argument(
        "--num_heatmap_views",
        type=int,
        default=8,
        help="Number of views to sample from heatmap.",
    )
    parser.add_argument(
        "--min_angle_dist_deg",
        type=float,
        default=20.0,
        help="Minimum angular separation (degrees) between heatmap-sampled candidate views.",
    )
    parser.add_argument(
        "--constrain_candidates_to_scene",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require candidate camera centers to lie inside the scene OBB (largest OBB in atomic_geometry.pt).",
    )
    parser.add_argument(
        "--scene_obb_margin_ratio",
        type=float,
        default=0.005,
        help="Shrink the scene OBB by (max_extent * ratio) when validating candidate cameras.",
    )
    parser.add_argument(
        "--render_instance_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When rendering candidate views, render only gaussians that belong to the instance (default: disabled).",
    )
    parser.add_argument(
        "--render_res", type=int, default=1024, help="Resolution for candidate view rendering."
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=70.0,
        help="Vertical field-of-view (degrees) for candidate views.",
    )
    parser.add_argument(
        "--render_flip_lr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Horizontally flip rendered assets; recorded in exported camera metadata.",
    )
    parser.add_argument(
        "--check_occlusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute per-view occlusion ratio using a mask-based white/black rendering trick.",
    )
    parser.add_argument(
        "--check_self_occlusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calculate point visibility ratio to filter bad viewing angles (e.g. back-facing).",
    )
    parser.add_argument(
        "--max_occlusion_ratio",
        type=float,
        default=0.7,
        help="Drop candidate views whose occlusion ratio exceeds this threshold (only if --check_occlusion).",
    )
    parser.add_argument(
        "--occlusion_alpha_threshold",
        type=float,
        default=0.8,
        help="Alpha threshold used when computing the ideal (instance-only) visible area.",
    )
    parser.add_argument(
        "--occlusion_white_threshold",
        type=float,
        default=0.5,
        help="White threshold used when counting visible pixels in the black/white scene render.",
    )
    parser.add_argument(
        "--export_topk",
        type=int,
        default=1,
        help=(
            "Export top-K ranked candidate view renders (+ masks when available) into "
            "cluster_result/candidate_views/{images,projected_mask}. "
            "Use 0 to disable exporting."
        ),
    )
    return parser.parse_args()


def discover_tracking_path(data_dir: Path) -> Path:
    candidates = [
        data_dir / "cluster_result" / "gauscluster_tracking_data.pt",
        data_dir / "gauscluster_tracking_data.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find gauscluster_tracking_data.pt under data_dir."
    )


def discover_instance_labels_knn_path(data_dir: Path) -> Optional[Path]:
    """
    Find the per-Gaussian instance label array generated by instascene_gauscluster.py.

    Prefers KNN-filled labels:
      - {data_dir}/cluster_result/instance_labels_knn.npy
      - {data_dir}/instance_labels_knn.npy
    """
    candidates = [
        data_dir / "cluster_result" / "instance_labels_knn.npy",
        data_dir / "instance_labels_knn.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=2)
def _load_instance_labels_knn(labels_path: str) -> np.ndarray:
    labels = np.load(labels_path, allow_pickle=False)
    labels = np.asarray(labels)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if labels.dtype != np.int32 and labels.dtype != np.int64:
        labels = labels.astype(np.int32, copy=False)
    return labels


def fibonacci_sphere(num_samples: int) -> np.ndarray:
    points = np.zeros((num_samples, 3), dtype=np.float32)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(num_samples):
        y = 1.0 - (i / float(num_samples - 1)) * 2.0
        radius = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points[i] = [x, y, z]
    return points


def heat_to_color(values: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    colors = cmap(values)[:, :4]
    return (colors * 255).astype(np.uint8)


def collect_camera_centers(
    view_data: List[Dict],
    scene_rotation: np.ndarray = None,
    floor_z: float = None,
) -> np.ndarray:
    centers = []
    for view in view_data:
        camtoworld = view["camtoworld"]
        if isinstance(camtoworld, torch.Tensor):
            camtoworld = camtoworld.cpu().numpy()
        center = camtoworld[:3, 3]
        if scene_rotation is not None:
            center = center @ scene_rotation.T
        if floor_z is not None:
            center = center.copy()
            center[2] -= float(floor_z)
        centers.append(center)
    return np.stack(centers, axis=0).astype(np.float32)


def apply_scene_transform(
    points: np.ndarray,
    scene_rotation: np.ndarray = None,
    floor_z: float = None,
) -> np.ndarray:
    if scene_rotation is not None:
        points = points @ scene_rotation.T
    if floor_z is not None:
        points = points.copy()
        points[:, 2] -= float(floor_z)
    return points


def build_camera_frustum_lines(
    camtoworld: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    depth: float,
    scene_rotation: np.ndarray = None,
    floor_z: float = None,
) -> List[np.ndarray]:
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    corners_px = np.array(
        [
            [0.0, 0.0],
            [float(width), 0.0],
            [float(width), float(height)],
            [0.0, float(height)],
        ],
        dtype=np.float32,
    )
    corners_cam = []
    for u, v in corners_px:
        x = (u - cx) / fx * depth
        y = (v - cy) / fy * depth
        z = depth
        corners_cam.append([x, y, z, 1.0])
    corners_cam = np.asarray(corners_cam, dtype=np.float32)
    origin_cam = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    corners_world = (camtoworld @ corners_cam.T).T[:, :3]
    origin_world = (camtoworld @ origin_cam.T).T[:, :3]

    corners_world = apply_scene_transform(corners_world, scene_rotation, floor_z)
    origin_world = apply_scene_transform(origin_world, scene_rotation, floor_z)

    lines = []
    for corner in corners_world:
        lines.append(np.stack([origin_world[0], corner], axis=0))
    for i in range(4):
        a = corners_world[i]
        b = corners_world[(i + 1) % 4]
        lines.append(np.stack([a, b], axis=0))
    return lines


def collect_visible_cameras(
    mask_dir: Path,
    view_data: List[Dict],
    instance_labels: List[int],
    top_visible: int,
) -> Tuple[List[int], List[int]]:
    visible_indices = []
    visible_counts = []
    labels = np.asarray([int(x) for x in instance_labels], dtype=np.int32).reshape(-1)
    for idx, view in enumerate(view_data):
        image_name = view["image_name"]
        mask_path = mask_dir / f"{image_name}.png"
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim > 2:
            mask = mask[..., 0]
        count = int(np.count_nonzero(np.isin(mask.astype(np.int32, copy=False), labels)))
        if count > 0:
            visible_indices.append(idx)
            visible_counts.append(count)
    if len(visible_counts) == 0:
        return [], []
    if top_visible <= 0 or top_visible >= len(visible_counts):
        return visible_indices, visible_counts

    weights = torch.tensor(visible_counts, dtype=torch.float32)
    sample_count = min(top_visible, len(visible_counts))
    sampled = torch.multinomial(weights, sample_count, replacement=False)
    sampled_indices = sampled.cpu().numpy().tolist()
    visible_indices = [visible_indices[i] for i in sampled_indices]
    visible_counts = [visible_counts[i] for i in sampled_indices]
    return visible_indices, visible_counts


def create_oriented_wireframe(
    center: np.ndarray,
    extent: np.ndarray,
    rotation: np.ndarray,
    color_rgba: np.ndarray,
    thickness_ratio: float,
    min_thickness: float,
) -> trimesh.Trimesh:
    """Build an OBB wireframe using 12 thin box meshes."""
    extent = np.maximum(extent, 1e-6)
    diag = float(np.linalg.norm(extent))
    thickness = max(diag * thickness_ratio, min_thickness)
    thickness = max(thickness, 1e-6)

    half = extent * 0.5
    hx, hy, hz = float(half[0]), float(half[1]), float(half[2])
    t = thickness
    sticks = []

    def add_stick(dim, trans):
        mat = tf.translation_matrix(trans)
        sticks.append(trimesh.creation.box(extents=dim, transform=mat))

    for dy in (-hy, hy):
        for dz in (-hz, hz):
            add_stick([extent[0], t, t], [0, dy, dz])
    for dx in (-hx, hx):
        for dz in (-hz, hz):
            add_stick([t, extent[1], t], [dx, 0, dz])
    for dx in (-hx, hx):
        for dy in (-hy, hy):
            add_stick([t, t, extent[2]], [dx, dy, 0])

    wireframe = trimesh.util.concatenate(sticks)
    wireframe.visual.face_colors = color_rgba

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    wireframe.apply_transform(transform)
    return wireframe


def compute_heatmap(
    instance_center: np.ndarray,
    camera_centers: np.ndarray,
    visible_indices: List[int],
    visible_counts: List[int],
    num_samples: int,
    sharpness: float,
    use_mask_weight: bool,
    aggregation_mode: str = "max",
) -> Tuple[np.ndarray, np.ndarray]:
    if len(visible_indices) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    cam_pos = camera_centers[visible_indices]
    vec = cam_pos - instance_center[None, :]
    norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6
    dirs = vec / norms

    sphere_points = fibonacci_sphere(num_samples)
    sim = sphere_points @ dirs.T
    sim = np.maximum(sim, 0.0)
    sim_sharp = sim ** float(sharpness)

    pixel_weights = np.ones(len(visible_indices), dtype=np.float32)
    if use_mask_weight:
        weights = np.asarray(visible_counts, dtype=np.float32)
        weights = np.log10(weights + 10.0)
        w_min = float(weights.min())
        w_max = float(weights.max())
        if w_max > w_min:
            weights = (weights - w_min) / (w_max - w_min)
        else:
            weights[:] = 1.0
        pixel_weights = 0.5 + 0.5 * weights

    mode = str(aggregation_mode).lower()
    if mode == "max":
        # Hybrid heatmap: quality (Max) * density (Sum).
        # - quality: "best view around this direction" (keeps the good peak).
        # - density: "how many views support this direction" (penalizes sparse backside).
        heat_quality = (sim_sharp * pixel_weights[None, :]).max(axis=1)

        heat_density = np.log1p(sim_sharp.sum(axis=1))
        if heat_density.max() > 0:
            heat_density = heat_density / heat_density.max()
        heat_density = 0.2 + 0.8 * heat_density

        heat = heat_quality * heat_density
        if heat.max() > 0:
            heat = heat / heat.max()
    elif mode == "sum":
        heat = (sim_sharp * pixel_weights[None, :]).sum(axis=1)
        if heat.max() > 0:
            heat = heat / heat.max()
    else:
        raise ValueError(f"Unknown aggregation_mode: {aggregation_mode}")
    return sphere_points, heat


def estimate_scene_scale(camera_centers: np.ndarray) -> float:
    """
    Estimate a scene scale from camera center distribution (post scene transform).
    Returns an approximate scene diagonal length.
    """
    if camera_centers.size == 0:
        return 0.0
    mins = camera_centers.min(axis=0)
    maxs = camera_centers.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    return diag


def calculate_occlusion_by_mask(
    view: Dict,
    gaussians_all: Tuple[torch.Tensor, ...],
    instance_indices: np.ndarray,
    device: torch.device,
    alpha_threshold: float = 0.99,
    white_threshold: float = 0.5,
) -> float:
    """
    Occlusion ratio via silhouette rendering:

    - Ideal area: render instance-only, count pixels with alpha > alpha_threshold
    - Visible area: render full scene with instance=white, others=black; count white pixels
    - occlusion = 1 - visible/ideal
    """
    if rasterization is None:
        raise RuntimeError("gsplat rasterization is not available.")

    width = int(view["resolution"])
    height = int(view["resolution"])
    fov_rad = math.radians(float(view["fov"]))
    focal = (float(height) / 2.0) / math.tan(fov_rad / 2.0)
    K = torch.tensor(
        [
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    c2w = torch.from_numpy(np.asarray(view["c2w"], dtype=np.float32)).to(
        device=device, dtype=torch.float32
    )
    viewmats = torch.linalg.inv(c2w).unsqueeze(0)

    means, quats, scales, opacities, _, _ = gaussians_all
    idx_tensor = torch.from_numpy(np.asarray(instance_indices, dtype=np.int64)).to(device)

    means_inst = means.index_select(0, idx_tensor)
    quats_inst = quats.index_select(0, idx_tensor)
    scales_inst = scales.index_select(0, idx_tensor)
    opacities_inst = opacities.index_select(0, idx_tensor)

    colors_inst_white = torch.ones(
        (means_inst.shape[0], 3), device=device, dtype=torch.float32
    )
    with torch.no_grad():
        _, render_alphas_inst, _ = rasterization(
            means=means_inst,
            quats=quats_inst,
            scales=scales_inst,
            opacities=opacities_inst,
            colors=colors_inst_white,
            viewmats=viewmats,
            Ks=K,
            width=width,
            height=height,
            sh_degree=None,
            render_mode="RGB",
            packed=False,
        )
    alpha_map = render_alphas_inst[0, ..., 0]
    ideal_pixels = int((alpha_map > float(alpha_threshold)).sum().item())
    if ideal_pixels <= 0:
        return 1.0

    num_total = int(means.shape[0])
    colors_scene = torch.zeros((num_total, 3), device=device, dtype=torch.float32)
    colors_scene.index_fill_(0, idx_tensor, 1.0)

    with torch.no_grad():
        render_colors_scene, render_alphas_scene, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors_scene,
            viewmats=viewmats,
            Ks=K,
            width=width,
            height=height,
            sh_degree=None,
            render_mode="RGB",
            packed=False,
        )
    rgb = render_colors_scene[0]
    alpha_scene = render_alphas_scene[0, ..., 0]
    rgb_max = torch.amax(rgb, dim=-1)
    visible_pixels = int(
        ((alpha_scene > float(alpha_threshold)) & (rgb_max > float(white_threshold)))
        .sum()
        .item()
    )

    ratio = 1.0 - (float(visible_pixels) / float(ideal_pixels))
    return float(np.clip(ratio, 0.0, 1.0))


def calculate_point_visibility_ratio(
    view: Dict,
    gaussians_inst: Tuple[torch.Tensor, ...],
    device: torch.device,
    obb_extent: np.ndarray,
    scale_ratio: float = 0.02,
) -> float:
    """
    Self-occlusion check: estimate how much of the instance surface is visible in this view.
    """
    width = int(view["resolution"])
    height = int(view["resolution"])

    fov_rad = math.radians(float(view["fov"]))
    focal = (float(height) / 2.0) / math.tan(fov_rad / 2.0)
    K = torch.tensor(
        [
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    rendered_depth = render_depth(
        gaussians=gaussians_inst,
        c2w_np=view["c2w"],
        K=K,
        width=width,
        height=height,
        device=device,
    )

    means = gaussians_inst[0].to(device=device, dtype=torch.float32)
    num_points = int(means.shape[0])
    if num_points == 0:
        return 0.0

    c2w = torch.from_numpy(np.asarray(view["c2w"], dtype=np.float32)).to(
        device=device, dtype=torch.float32
    )
    w2c = torch.linalg.inv(c2w)
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    points_cam = (R @ means.T).T + T

    z_vals = points_cam[:, 2]
    valid_mask = z_vals > 0.05

    x_vals = points_cam[:, 0]
    y_vals = points_cam[:, 1]
    u_vals = (x_vals * focal / (z_vals + 1e-6)) + (width / 2.0)
    v_vals = (y_vals * focal / (z_vals + 1e-6)) + (height / 2.0)

    valid_mask &= (u_vals >= 0.0) & (u_vals < float(width) - 0.5)
    valid_mask &= (v_vals >= 0.0) & (v_vals < float(height) - 0.5)

    valid_idx = torch.nonzero(valid_mask).squeeze(-1)
    if valid_idx.numel() == 0:
        return 0.0

    u_valid = u_vals[valid_idx].long()
    v_valid = v_vals[valid_idx].long()
    z_point = z_vals[valid_idx]
    z_render = rendered_depth[v_valid, u_valid]

    diag = float(np.linalg.norm(np.asarray(obb_extent, dtype=np.float32)))
    threshold = max(diag * float(scale_ratio), 0.005)
    is_visible = (z_render > 0.0) & (z_point <= (z_render + float(threshold)))
    visible_count = int(is_visible.sum().item())
    return float(visible_count) / float(num_points)


def find_largest_scene_obb(
    atomic: Dict,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Use the largest instance OBB in atomic_geometry.pt as a proxy for the scene bounds.

    Returns (center, extent, rotation) in the same coordinate system as atomic.
    """
    best: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    best_volume = -1.0
    for _, info in atomic.items():
        if not isinstance(info, dict):
            continue
        if "obb_center" not in info or "obb_extent" not in info:
            continue
        center = np.asarray(info["obb_center"], dtype=np.float32).reshape(3)
        extent = np.asarray(info["obb_extent"], dtype=np.float32).reshape(3)
        rotation = np.asarray(info.get("obb_rotation", np.eye(3)), dtype=np.float32).reshape(
            3, 3
        )
        if np.any(extent <= 1e-6) or not np.all(np.isfinite(extent)):
            continue
        volume = float(extent[0] * extent[1] * extent[2])
        if volume > best_volume:
            best_volume = volume
            best = (center, extent, rotation)
    return best


def point_inside_obb(
    point: np.ndarray,
    center: np.ndarray,
    extent: np.ndarray,
    rotation: np.ndarray,
    margin: float = 0.0,
) -> bool:
    point = np.asarray(point, dtype=np.float32).reshape(3)
    center = np.asarray(center, dtype=np.float32).reshape(3)
    extent = np.asarray(extent, dtype=np.float32).reshape(3)
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    half = extent * 0.5 - float(margin)
    if np.any(half <= 1e-6):
        return False
    local = rotation.T @ (point - center)
    return bool(np.all(np.abs(local) <= half + 1e-6))


def intrinsics_from_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    fov_rad = math.radians(float(fov_deg))
    focal = (float(height) / 2.0) / math.tan(fov_rad / 2.0)
    return np.array(
        [
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_gaussians_from_ckpt(
    ckpt_path: str, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    if "splats" not in payload:
        raise KeyError(f"Checkpoint missing 'splats' key: {ckpt_path}")
    splats = payload["splats"]

    means = splats["means"]
    quats = F.normalize(splats["quats"], dim=-1)
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    sh0 = splats["sh0"]
    shn = splats["shN"]
    colors = torch.cat([sh0, shn], dim=-2)
    sh_degree = int(np.sqrt(colors.shape[-2]) - 1)
    return means, quats, scales, opacities, colors, sh_degree


def load_instance_gaussian_ids(data_dir: Path, instance_id: int) -> Optional[np.ndarray]:
    """
    Load Gaussian indices that belong to an instance.

    Preference order:
      1) instance_labels_knn.npy (per-Gaussian label array; includes KNN-filled unlabeled points)
      2) gauscluster_tracking_data.pt total_point_ids_list (legacy)

    Returns None if tracking file or the mapping is unavailable.
    """
    labels_path = discover_instance_labels_knn_path(data_dir)
    if labels_path is not None:
        labels = _load_instance_labels_knn(str(labels_path))
        if labels.size == 0:
            return np.empty((0,), dtype=np.int64)
        ids = np.nonzero(labels == int(instance_id))[0]
        return np.asarray(ids, dtype=np.int64).reshape(-1)

    try:
        tracking_path = discover_tracking_path(data_dir)
    except FileNotFoundError:
        return None

    tracking = torch.load(tracking_path, map_location="cpu", weights_only=False)
    total_point_ids_list = tracking.get("total_point_ids_list", None)
    if total_point_ids_list is None:
        return None
    if not isinstance(total_point_ids_list, (list, tuple)):
        return None
    if instance_id < 0 or instance_id >= len(total_point_ids_list):
        return None

    ids = total_point_ids_list[instance_id]
    if ids is None:
        return np.empty((0,), dtype=np.int64)
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()
    return np.asarray(ids, dtype=np.int64).reshape(-1)


def load_instance_gaussian_ids_multi(
    data_dir: Path, instance_ids: List[int]
) -> Optional[np.ndarray]:
    """
    Load and union Gaussian indices for multiple instances.

    Returns None if tracking file or the mapping is unavailable.
    """
    labels_path = discover_instance_labels_knn_path(data_dir)
    if labels_path is not None:
        labels = _load_instance_labels_knn(str(labels_path))
        if labels.size == 0:
            return np.empty((0,), dtype=np.int64)
        wanted = np.asarray([int(i) for i in instance_ids], dtype=labels.dtype).reshape(-1)
        ids = np.nonzero(np.isin(labels, wanted))[0]
        return np.asarray(ids, dtype=np.int64).reshape(-1)

    all_ids: List[np.ndarray] = []
    for instance_id in instance_ids:
        ids = load_instance_gaussian_ids(data_dir=data_dir, instance_id=int(instance_id))
        if ids is None:
            return None
        if ids.size == 0:
            continue
        all_ids.append(np.asarray(ids, dtype=np.int64).reshape(-1))

    if not all_ids:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(all_ids, axis=0))


def filter_gaussians_by_ids(
    gaussians: Tuple[torch.Tensor, ...], gaussian_ids: np.ndarray
) -> Tuple[torch.Tensor, ...]:
    means, quats, scales, opacities, colors, sh_degree = gaussians
    if gaussian_ids.size == 0:
        raise ValueError("Instance gaussian id list is empty.")
    if gaussian_ids.min() < 0 or gaussian_ids.max() >= int(means.shape[0]):
        raise ValueError("Instance gaussian ids out of range for checkpoint.")
    idx = torch.from_numpy(gaussian_ids).to(device=means.device)
    return (
        means.index_select(0, idx),
        quats.index_select(0, idx),
        scales.index_select(0, idx),
        opacities.index_select(0, idx),
        colors.index_select(0, idx),
        sh_degree,
    )


def render_camera_view(
    gaussians: Tuple[torch.Tensor, ...],
    c2w_np: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    device: torch.device,
) -> np.ndarray:
    if not HAS_GSPLAT or rasterization is None:
        raise RuntimeError("gsplat is not available; cannot render.")

    means, quats, scales, opacities, colors, sh_degree = gaussians
    means = means.to(device=device, dtype=torch.float32)
    quats = quats.to(device=device, dtype=torch.float32)
    scales = scales.to(device=device, dtype=torch.float32)
    opacities = opacities.to(device=device, dtype=torch.float32)
    colors = colors.to(device=device, dtype=torch.float32)

    c2w = torch.from_numpy(c2w_np).to(device=device, dtype=torch.float32)
    viewmats = torch.linalg.inv(c2w).unsqueeze(0)

    fov_rad = math.radians(float(fov_deg))
    focal = (float(height) / 2.0) / math.tan(fov_rad / 2.0)
    K = torch.tensor(
        [
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    return render_camera_view_with_K(
        gaussians=gaussians,
        c2w_np=c2w_np,
        K=K,
        width=width,
        height=height,
        device=device,
    )


def render_camera_view_with_K(
    gaussians: Tuple[torch.Tensor, ...],
    c2w_np: np.ndarray,
    K: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
) -> np.ndarray:
    if not HAS_GSPLAT or rasterization is None:
        raise RuntimeError("gsplat is not available; cannot render.")

    means, quats, scales, opacities, colors, sh_degree = gaussians
    means = means.to(device=device, dtype=torch.float32)
    quats = quats.to(device=device, dtype=torch.float32)
    scales = scales.to(device=device, dtype=torch.float32)
    opacities = opacities.to(device=device, dtype=torch.float32)
    colors = colors.to(device=device, dtype=torch.float32)

    c2w = torch.from_numpy(c2w_np).to(device=device, dtype=torch.float32)
    viewmats = torch.linalg.inv(c2w).unsqueeze(0)

    with torch.no_grad():
        render_colors, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=K,
            width=int(width),
            height=int(height),
            sh_degree=sh_degree,
            render_mode="RGB",
            packed=False,
        )

    img = render_colors[0].clamp(0, 1).cpu().numpy()
    return (img * 255.0).astype(np.uint8)


def render_depth(
    gaussians: Tuple[torch.Tensor, ...],
    c2w_np: np.ndarray,
    K: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Render a depth map (H, W) and prefer median depth (more surface-like).
    Falls back to expected depth if median depth is unavailable.
    """
    if not HAS_GSPLAT or rasterization is None:
        return torch.zeros((height, width), device=device, dtype=torch.float32)

    means, quats, scales, opacities, colors, sh_degree = gaussians
    means = means.to(device=device)
    quats = quats.to(device=device)
    scales = scales.to(device=device)
    opacities = opacities.to(device=device)
    colors = colors.to(device=device)

    c2w = torch.from_numpy(np.asarray(c2w_np, dtype=np.float32)).to(
        device=device, dtype=torch.float32
    )
    viewmats = torch.linalg.inv(c2w).unsqueeze(0)

    with torch.no_grad():
        render_colors, _, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=K,
            width=int(width),
            height=int(height),
            sh_degree=sh_degree,
            render_mode="RGB+ED",
            packed=False,
        )

    render_median = meta.get("render_median", None) if isinstance(meta, dict) else None
    if isinstance(render_median, torch.Tensor) and render_median.numel() > 0:
        return render_median[0, ..., 0]
    return render_colors[0, ..., -1]


def _depth_to_vis_rgb(depth: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W) depth map to an RGB visualization.
    Uses robust percentiles for normalization and a colormap for display.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not bool(valid.any()):
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    vals = depth[valid]
    vmin = float(np.percentile(vals, 2.0))
    vmax = float(np.percentile(vals, 98.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin + 1e-6:
        vmin = float(vals.min())
        vmax = float(vals.max() if vals.max() > vals.min() + 1e-6 else vals.min() + 1.0)

    norm = (depth - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    norm_u8[~valid] = 0

    bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb


def render_alpha_map(
    gaussians: Tuple[torch.Tensor, ...],
    c2w_np: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    device: torch.device,
) -> np.ndarray:
    """
    Render an alpha map (H, W) in [0, 1] for the given gaussians and camera pose.
    """
    if not HAS_GSPLAT or rasterization is None:
        raise RuntimeError("gsplat is not available; cannot render alpha.")

    means, quats, scales, opacities, _, _ = gaussians
    means = means.to(device=device, dtype=torch.float32)
    quats = quats.to(device=device, dtype=torch.float32)
    scales = scales.to(device=device, dtype=torch.float32)
    opacities = opacities.to(device=device, dtype=torch.float32)

    c2w = torch.from_numpy(np.asarray(c2w_np, dtype=np.float32)).to(
        device=device, dtype=torch.float32
    )
    viewmats = torch.linalg.inv(c2w).unsqueeze(0)

    fov_rad = math.radians(float(fov_deg))
    focal = (float(height) / 2.0) / math.tan(fov_rad / 2.0)
    K = torch.tensor(
        [
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    colors_white = torch.ones((int(means.shape[0]), 3), device=device, dtype=torch.float32)
    with torch.no_grad():
        _, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors_white,
            viewmats=viewmats,
            Ks=K,
            width=int(width),
            height=int(height),
            sh_degree=None,
            render_mode="RGB",
            packed=False,
        )

    alpha = render_alphas[0, ..., 0].clamp(0, 1).detach().cpu().numpy()
    return alpha.astype(np.float32, copy=False)


def render_visible_instance_mask(
    gaussians_all: Tuple[torch.Tensor, ...],
    instance_indices: np.ndarray,
    c2w_np: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    device: torch.device,
    *,
    alpha_threshold: float,
    white_threshold: float,
) -> np.ndarray:
    """
    Render a binary visible-instance mask (H, W) as uint8 {0,255} using the full scene:

    - Render full scene with instance=white, others=black (colors as constants, sh_degree=None).
    - Visible instance pixels are those with alpha > alpha_threshold and white > white_threshold.

    This accounts for occlusion by other gaussians.
    """
    if not HAS_GSPLAT or rasterization is None:
        raise RuntimeError("gsplat is not available; cannot render visible instance mask.")

    means, quats, scales, opacities, _, _ = gaussians_all
    means = means.to(device=device, dtype=torch.float32)
    quats = quats.to(device=device, dtype=torch.float32)
    scales = scales.to(device=device, dtype=torch.float32)
    opacities = opacities.to(device=device, dtype=torch.float32)

    c2w = torch.from_numpy(np.asarray(c2w_np, dtype=np.float32)).to(
        device=device, dtype=torch.float32
    )
    viewmats = torch.linalg.inv(c2w).unsqueeze(0)

    fov_rad = math.radians(float(fov_deg))
    focal = (float(height) / 2.0) / math.tan(fov_rad / 2.0)
    K = torch.tensor(
        [
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    idx_tensor = torch.from_numpy(np.asarray(instance_indices, dtype=np.int64)).to(device)
    colors_scene = torch.zeros((int(means.shape[0]), 3), device=device, dtype=torch.float32)
    colors_scene.index_fill_(0, idx_tensor, 1.0)

    with torch.no_grad():
        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors_scene,
            viewmats=viewmats,
            Ks=K,
            width=int(width),
            height=int(height),
            sh_degree=None,
            render_mode="RGB",
            packed=False,
        )

    rgb = render_colors[0]
    alpha = render_alphas[0, ..., 0]
    rgb_max = torch.amax(rgb, dim=-1)
    visible = (alpha > float(alpha_threshold)) & (rgb_max > float(white_threshold))
    return (visible.detach().cpu().numpy().astype(np.uint8) * 255).astype(np.uint8, copy=False)


def _normalize_np(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def get_lookat_matrix_opencv(
    cam_pos: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),
) -> np.ndarray:
    """
    Build a COLMAP/OpenCV-style camera-to-world matrix:
    x: right, y: down, z: forward (looking direction).
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = _normalize_np(target - cam_pos)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = _normalize_np(right)
    up_cam = _normalize_np(np.cross(right, forward))
    down = -up_cam

    mat = np.eye(4, dtype=np.float32)
    mat[:3, 0] = right
    mat[:3, 1] = down
    mat[:3, 2] = forward
    mat[:3, 3] = cam_pos
    return mat


def get_obb_corners_local(extent: np.ndarray) -> np.ndarray:
    extent = np.asarray(extent, dtype=np.float32)
    x, y, z = float(extent[0] / 2.0), float(extent[1] / 2.0), float(extent[2] / 2.0)
    return np.array(
        [
            [x, y, z],
            [x, y, -z],
            [x, -y, z],
            [x, -y, -z],
            [-x, y, z],
            [-x, y, -z],
            [-x, -y, z],
            [-x, -y, -z],
        ],
        dtype=np.float32,
    )


class _DSU:
    def __init__(self, nodes: List[int]):
        self.parent = {int(n): int(n) for n in nodes}

    def find(self, x: int) -> int:
        px = self.parent.get(int(x), int(x))
        if px != x:
            self.parent[int(x)] = self.find(px)
        return self.parent.get(int(x), int(x))

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(int(a)), self.find(int(b))
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def build_merge_groups(instance_ids: List[int], merge_pairs: List[Tuple[int, int]]) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Build merge groups from the supplied pairs, returning:
      - groups: representative_id -> member_ids
      - mapping: instance_id -> representative_id

    Representative is always the smallest id in a group.
    """
    base_set = {int(i) for i in instance_ids}
    nodes = sorted(
        base_set
        | {int(a) for a, _ in merge_pairs}
        | {int(b) for _, b in merge_pairs}
    )
    if not nodes:
        return {}, {}
    dsu = _DSU(nodes)
    for a, b in merge_pairs:
        dsu.union(int(a), int(b))

    groups: Dict[int, List[int]] = {}
    for idx in nodes:
        root = dsu.find(int(idx))
        groups.setdefault(root, []).append(int(idx))

    mapping: Dict[int, int] = {}
    rep_to_members: Dict[int, List[int]] = {}
    for members in groups.values():
        base_members = [m for m in members if m in base_set]
        rep = int(min(base_members)) if base_members else int(min(members))
        rep_to_members[rep] = sorted(set(members))
        for m in members:
            mapping[int(m)] = rep
    return rep_to_members, mapping


def merge_instance_obbs(
    atomic: Dict, member_ids: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge multiple instance OBBs into a single OBB aligned with the smallest-id member's rotation.
    Returns (center, extent, rotation) in the same coordinate system as atomic.
    """
    if not member_ids:
        raise ValueError("member_ids is empty.")

    rep_candidates = [int(i) for i in member_ids if int(i) in atomic]
    if not rep_candidates:
        raise KeyError("No member_ids exist in atomic.")
    rep_id = int(min(rep_candidates))
    rep_info = atomic[int(rep_id)]
    rep_rot = np.asarray(rep_info.get("obb_rotation", np.eye(3)), dtype=np.float32).reshape(3, 3)

    corners_world: List[np.ndarray] = []
    for mid in member_ids:
        info = atomic.get(int(mid), None)
        if not isinstance(info, dict):
            continue
        center = np.asarray(info["obb_center"], dtype=np.float32).reshape(3)
        extent = np.asarray(info["obb_extent"], dtype=np.float32).reshape(3)
        rot = np.asarray(info.get("obb_rotation", np.eye(3)), dtype=np.float32).reshape(3, 3)
        local = get_obb_corners_local(extent)  # [8, 3]
        corners = (rot @ local.T).T + center[None, :]
        corners_world.append(corners)

    if not corners_world:
        center = np.asarray(rep_info["obb_center"], dtype=np.float32).reshape(3)
        extent = np.asarray(rep_info["obb_extent"], dtype=np.float32).reshape(3)
        return center, extent, rep_rot

    all_corners = np.concatenate(corners_world, axis=0)  # [N, 3]
    all_local = (rep_rot.T @ all_corners.T).T
    mins = all_local.min(axis=0)
    maxs = all_local.max(axis=0)
    center_local = (mins + maxs) * 0.5
    extent = np.maximum(maxs - mins, 1e-6).astype(np.float32)
    center = (rep_rot @ center_local.reshape(3, 1)).reshape(3).astype(np.float32)
    return center, extent, rep_rot


def calculate_tight_distance(
    points_local: np.ndarray,
    cam_dir_local: np.ndarray,
    fov_rad: float,
    padding: float = 1.1,
) -> float:
    """
    Compute the minimal camera distance so that all local points fit in the frustum.

    Assumptions:
    - target is at the local origin,
    - camera is placed at cam_pos = cam_dir_local * dist,
    - camera looks towards the origin (OpenCV-style +Z forward),
    - square viewport so horizontal and vertical FOV match.
    """
    cam_dir_local = _normalize_np(np.asarray(cam_dir_local, dtype=np.float32))
    forward = _normalize_np(-cam_dir_local)
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up_world)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = _normalize_np(right)
    up_cam = _normalize_np(np.cross(right, forward))
    down = -up_cam

    R_cw = np.stack([right, down, forward], axis=1).astype(np.float32)  # [3, 3]
    points_cam0 = (R_cw.T @ points_local.T).T  # dist=0, translation handled separately

    tan_half = math.tan(float(fov_rad) / 2.0)
    x_req = np.abs(points_cam0[:, 0]) / (tan_half + 1e-12) - points_cam0[:, 2]
    y_req = np.abs(points_cam0[:, 1]) / (tan_half + 1e-12) - points_cam0[:, 2]
    z_req = -points_cam0[:, 2] + 1e-3
    dist = float(np.max(np.maximum.reduce([x_req, y_req, z_req])))
    dist = max(dist, 0.1)
    return dist * float(padding)


def generate_adaptive_views(
    obb_center: np.ndarray,
    obb_extent: np.ndarray,
    obb_rotation: np.ndarray,
    fov: float = 70.0,
    resolution: int = 1024,
    scene_obb: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    scene_obb_margin: float = 0.0,
    min_camera_distance: Optional[float] = None,
) -> List[Dict]:
    extents = np.asarray(obb_extent, dtype=np.float32)
    max_idx = int(np.argmax(extents[:2]))
    length = float(extents[max_idx])
    width = float(extents[1 - max_idx])
    aspect_ratio = length / (width + 1e-4)
    is_long = aspect_ratio > 2.0

    if is_long:
        print(
            f"[Candidates] Detected long object (AR={aspect_ratio:.2f}). Using broad-side bias."
        )
        offset = 25.0
        base_angles = [90.0, 270.0] if max_idx == 0 else [0.0, 180.0]
        preferred_azimuths: List[float] = []
        for base in base_angles:
            preferred_azimuths.extend([base - offset, base, base + offset])
        elevation = 15.0
    else:
        preferred_azimuths = np.arange(0.0, 360.0, 45.0).tolist()
        elevation = 20.0

    target_count = len(preferred_azimuths)
    fallback_azimuths = np.arange(0.0, 360.0, 10.0).tolist()
    azimuths: List[float] = []
    seen = set()
    for a in preferred_azimuths + fallback_azimuths:
        key = int(round(a)) % 360
        if key in seen:
            continue
        seen.add(key)
        azimuths.append(float(a))

    fov_rad = math.radians(float(fov))
    corners_local = get_obb_corners_local(extents)
    views: List[Dict] = []

    for az in azimuths:
        az_rad = math.radians(float(az))
        el_rad = math.radians(float(elevation))

        cx = math.cos(el_rad) * math.cos(az_rad)
        cy = math.cos(el_rad) * math.sin(az_rad)
        cz = math.sin(el_rad)
        cam_dir_local = _normalize_np(np.array([cx, cy, cz], dtype=np.float32))

        dist = calculate_tight_distance(
            corners_local, cam_dir_local, fov_rad, padding=1.1
        )
        if min_camera_distance is not None and float(min_camera_distance) > 0:
            dist = max(float(dist), float(min_camera_distance))
        cam_pos_local = cam_dir_local * dist
        cam_pos_world = obb_rotation @ cam_pos_local + obb_center
        if scene_obb is not None:
            scene_center, scene_extent, scene_rot = scene_obb
            if not point_inside_obb(
                cam_pos_world,
                center=scene_center,
                extent=scene_extent,
                rotation=scene_rot,
                margin=scene_obb_margin,
            ):
                continue
        c2w = get_lookat_matrix_opencv(cam_pos_world, obb_center)

        i = len(views)
        views.append(
            {
                "name": f"view_{i:02d}_az{int(round(az))}_el{int(round(elevation))}",
                "c2w": c2w,
                "fov": float(fov),
                "resolution": int(resolution),
                "heatmap_score": 0.5,
            }
        )
        if len(views) >= target_count:
            break

    return views


def generate_heatmap_based_views(
    sphere_points: np.ndarray,
    heat_values: np.ndarray,
    obb_center: np.ndarray,
    obb_extent: np.ndarray,
    obb_rotation: np.ndarray,
    num_views: int = 6,
    fov: float = 70.0,
    resolution: int = 1024,
    min_heat_threshold: float = 0.1,
    min_angle_dist_deg: float = 20.0,
    scene_obb: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    scene_obb_margin: float = 0.0,
    min_camera_distance: Optional[float] = None,
) -> List[Dict]:
    """
    Select high-visibility directions from the heatmap and convert them into camera poses.

    Core logic:
    - filter low-heat directions
    - greedy select top heat with angular diversity
    - compute tight-fit distance using OBB corners in local frame
    """
    if sphere_points.size == 0 or heat_values.size == 0 or num_views <= 0:
        return []

    sphere_points = np.asarray(sphere_points, dtype=np.float32)
    heat_values = np.asarray(heat_values, dtype=np.float32)
    if sphere_points.shape[0] != heat_values.shape[0]:
        raise ValueError(
            f"sphere_points/heat_values length mismatch: {sphere_points.shape[0]} vs {heat_values.shape[0]}"
        )

    valid = heat_values >= float(min_heat_threshold)
    if not np.any(valid):
        return []

    candidates = sphere_points[valid]
    heats = heat_values[valid]
    order = np.argsort(-heats)
    candidates = candidates[order]
    heats = heats[order]

    fov_rad = math.radians(float(fov))
    corners_local = get_obb_corners_local(np.asarray(obb_extent, dtype=np.float32))
    obb_rotation = np.asarray(obb_rotation, dtype=np.float32)
    obb_center = np.asarray(obb_center, dtype=np.float32)

    min_cos = math.cos(math.radians(float(min_angle_dist_deg)))
    selected_dirs: List[np.ndarray] = []
    selected_heats: List[float] = []

    def _try_add(direction_world: np.ndarray, heat: float, enforce_angle: bool) -> bool:
        direction_world = _normalize_np(direction_world)
        if not selected_dirs:
            pass
        elif enforce_angle:
            dots = np.array(
                [float(np.dot(direction_world, d)) for d in selected_dirs], dtype=np.float32
            )
            if not np.all(dots <= min_cos):
                return False

        cam_dir_world = direction_world
        cam_dir_local = _normalize_np(obb_rotation.T @ cam_dir_world)
        dist = calculate_tight_distance(
            points_local=corners_local,
            cam_dir_local=cam_dir_local,
            fov_rad=fov_rad,
            padding=1.1,
        )
        if min_camera_distance is not None and float(min_camera_distance) > 0:
            dist = max(float(dist), float(min_camera_distance))
        cam_pos_world = obb_rotation @ (cam_dir_local * dist) + obb_center
        if scene_obb is not None:
            scene_center, scene_extent, scene_rot = scene_obb
            if not point_inside_obb(
                cam_pos_world,
                center=scene_center,
                extent=scene_extent,
                rotation=scene_rot,
                margin=scene_obb_margin,
            ):
                return False

        selected_dirs.append(direction_world)
        selected_heats.append(float(heat))
        return True

    for direction, heat in zip(candidates, heats):
        if _try_add(direction, float(heat), enforce_angle=True) and len(selected_dirs) >= num_views:
            break

    if len(selected_dirs) < num_views:
        for direction, heat in zip(candidates, heats):
            if len(selected_dirs) >= num_views:
                break
            direction = _normalize_np(direction)
            if selected_dirs:
                if (
                    np.max(
                        np.array(
                            [float(np.dot(direction, d)) for d in selected_dirs],
                            dtype=np.float32,
                        )
                    )
                    > 0.9999
                ):
                    continue
            if _try_add(direction, float(heat), enforce_angle=False):
                continue

    views: List[Dict] = []
    for i, (dir_world, heat) in enumerate(zip(selected_dirs, selected_heats)):
        cam_dir_world = _normalize_np(np.asarray(dir_world, dtype=np.float32))
        cam_dir_local = _normalize_np(obb_rotation.T @ cam_dir_world)
        dist = calculate_tight_distance(
            points_local=corners_local,
            cam_dir_local=cam_dir_local,
            fov_rad=fov_rad,
            padding=1.1,
        )
        if min_camera_distance is not None and float(min_camera_distance) > 0:
            dist = max(float(dist), float(min_camera_distance))
        cam_pos_world = obb_rotation @ (cam_dir_local * dist) + obb_center
        c2w = get_lookat_matrix_opencv(cam_pos_world, obb_center)
        views.append(
            {
                "name": f"heat_{i:02d}_h{heat:.2f}",
                "c2w": c2w,
                "fov": float(fov),
                "resolution": int(resolution),
                "heatmap_score": float(heat),
            }
        )
    return views


def find_checkpoint_heuristic(data_dir: Path) -> Optional[str]:
    def _score(path: Path) -> Tuple[int, int, str]:
        name = path.name.lower()
        keyword = 0
        if "final" in name:
            keyword += 3
        if "splats" in name:
            keyword += 2
        if "ckpt" in name or "checkpoint" in name:
            keyword += 1

        step = 0
        for token in name.replace(".", "_").split("_"):
            if token.isdigit():
                step = max(step, int(token))
        return (keyword, step, str(path))

    candidates: List[Path] = []
    for folder in [
        data_dir,
        data_dir / "checkpoints",
        data_dir / "ckpts",
        data_dir / "output",
        data_dir / "cluster_result",
    ]:
        if folder.exists() and folder.is_dir():
            candidates.extend(sorted(folder.glob("*.pt")))

    if not candidates:
        return None

    candidates.sort(key=_score, reverse=True)

    banned_names = {
        "gauscluster_tracking_data.pt",
        "atomic_geometry.pt",
        "gauscluster_input.pt",
    }
    candidates = [c for c in candidates if c.name not in banned_names]

    for candidate in candidates:
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        splats = payload.get("splats", None)
        if not isinstance(splats, dict):
            continue
        required = {"means", "quats", "scales", "opacities", "sh0", "shN"}
        if required.issubset(set(splats.keys())):
            return str(candidate)

    return str(candidates[0]) if candidates else None


def _alignment_inverse_transform(
    scene_rotation: Optional[np.ndarray], floor_z: Optional[float]
) -> np.ndarray:
    """
    Build T_o_a: aligned -> original, given the alignment used by prepare_atomic_geometry.py.
    """
    T = np.eye(4, dtype=np.float32)
    if scene_rotation is not None:
        T[:3, :3] = np.asarray(scene_rotation, dtype=np.float32).T
    if floor_z is not None:
        shift_aligned = np.array([0.0, 0.0, float(floor_z)], dtype=np.float32)
        T[:3, 3] = (T[:3, :3] @ shift_aligned.reshape(3, 1)).reshape(3)
    return T


def _load_view_data(data_dir: Path) -> List[Dict]:
    try:
        tracking_path = discover_tracking_path(data_dir)
    except FileNotFoundError:
        print("[Heatmap] Warning: tracking data not found; skip heatmap.")
        return []

    tracking = torch.load(tracking_path, map_location="cpu", weights_only=False)
    view_data = tracking.get("view_data", [])
    return view_data if isinstance(view_data, list) else []


def _prepare_scene_obb(
    args: argparse.Namespace, atomic: Dict
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]], float]:
    if not args.constrain_candidates_to_scene:
        return None, 0.0

    scene_obb = find_largest_scene_obb(atomic)
    if scene_obb is None:
        print("[Candidates] Warning: scene OBB not found; skipping scene constraint.")
        return None, 0.0

    _, scene_extent, _ = scene_obb
    margin = float(np.max(scene_extent)) * float(args.scene_obb_margin_ratio)
    print(f"[Candidates] Using scene OBB (largest) with margin={margin:.4f}")
    return scene_obb, margin


def _prepare_gaussians_for_rendering(
    args: argparse.Namespace, data_dir: Path
) -> Tuple[Optional[str], Optional[Tuple[torch.Tensor, ...]], Optional[torch.device], Optional[int]]:
    ckpt_path = args.ckpt or find_checkpoint_heuristic(data_dir)
    if ckpt_path is None:
        print("[Render] No checkpoint provided/found; will skip rendering PNGs.")
        return None, None, None, None
    if not HAS_GSPLAT:
        print("[Render] Skipping rendering (gsplat missing).")
        return ckpt_path, None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Render] Using device: {device}")
    print(f"[Gaussians] Loading checkpoint: {ckpt_path}")
    gaussians_all = load_gaussians_from_ckpt(ckpt_path, device)
    full_gaussian_count = int(gaussians_all[0].shape[0])
    return ckpt_path, gaussians_all, device, full_gaussian_count


def _maybe_build_heatmap_scene(
    *,
    args: argparse.Namespace,
    cluster_dir: Path,
    view_data: List[Dict],
    instance_ids: List[int],
    instance_center: np.ndarray,
    instance_extent: np.ndarray,
    instance_axes: np.ndarray,
    scene_rotation: Optional[np.ndarray],
    floor_z: Optional[float],
) -> Tuple[
    Optional[trimesh.Scene],
    np.ndarray,
    np.ndarray,
    float,
    Optional[float],
]:
    heatmap_sphere_points = np.empty((0, 3), dtype=np.float32)
    heatmap_values = np.empty((0,), dtype=np.float32)
    heatmap_scene: Optional[trimesh.Scene] = None
    heatmap_scene_diag = 0.0
    min_visible_train_dist: Optional[float] = None

    if len(view_data) == 0:
        return heatmap_scene, heatmap_sphere_points, heatmap_values, heatmap_scene_diag, None

    camera_centers = collect_camera_centers(
        view_data,
        scene_rotation=scene_rotation,
        floor_z=floor_z,
    )

    mask_dir = cluster_dir / "projected_sam"
    instance_labels = [int(i) + int(args.mask_id_offset) for i in instance_ids]
    visible_indices, visible_counts = collect_visible_cameras(
        mask_dir=mask_dir,
        view_data=view_data,
        instance_labels=instance_labels,
        top_visible=args.top_visible,
    )

    if len(visible_indices) > 0:
        dists = np.linalg.norm(
            camera_centers[visible_indices] - instance_center[None, :], axis=1
        )
        if dists.size > 0:
            min_visible_train_dist = float(np.min(dists))
            print(f"[Candidates] Min visible training distance: {min_visible_train_dist:.4f}")

    sphere_points, heat = compute_heatmap(
        instance_center=instance_center,
        camera_centers=camera_centers,
        visible_indices=visible_indices,
        visible_counts=visible_counts,
        num_samples=args.num_samples,
        sharpness=args.sharpness,
        use_mask_weight=args.use_mask_weight,
        aggregation_mode="max",
    )

    if sphere_points.shape[0] == 0:
        print("[Heatmap] Warning: no visible cameras for this instance; skip heatmap.")
        return heatmap_scene, heatmap_sphere_points, heatmap_values, heatmap_scene_diag, min_visible_train_dist

    heatmap_sphere_points = sphere_points
    heatmap_values = heat

    colors = heat_to_color(heat, cmap_name="turbo")
    scene = trimesh.Scene()

    base_radius = max(float(np.max(instance_extent) * 0.05), 1e-4)
    vis_threshold = 0.05
    spheres = []
    sphere_radius = float(np.max(instance_extent) * 0.7)
    for pt_unit, color, hval in zip(sphere_points, colors, heat):
        if hval < vis_threshold:
            continue
        scale_factor = (float(hval) ** 2) * 1.5 + 0.1
        current_radius = base_radius * scale_factor
        pt_world = pt_unit * sphere_radius + instance_center
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=current_radius)
        sphere.apply_translation(pt_world)
        sphere.visual.face_colors = np.array([color[0], color[1], color[2], 255], dtype=np.uint8)
        spheres.append(sphere)

    if spheres:
        scene.add_geometry(trimesh.util.concatenate(spheres))

    scene_diag = estimate_scene_scale(camera_centers=camera_centers)
    heatmap_scene_diag = float(scene_diag)
    frustum_depth = scene_diag * 0.01 if scene_diag > 1e-6 else float(np.max(instance_extent) * 0.12)
    frustum_color = np.array([255, 0, 0, 220], dtype=np.uint8)
    frustum_lines = []
    for idx in visible_indices:
        view = view_data[idx]
        camtoworld = view["camtoworld"]
        if isinstance(camtoworld, torch.Tensor):
            camtoworld = camtoworld.cpu().numpy()
        K = view["K"]
        if isinstance(K, torch.Tensor):
            K = K.cpu().numpy()
        width = int(view["width"])
        height = int(view["height"])
        frustum_lines.extend(
            build_camera_frustum_lines(
                camtoworld=camtoworld,
                K=K,
                width=width,
                height=height,
                depth=frustum_depth,
                scene_rotation=scene_rotation,
                floor_z=floor_z,
            )
        )

    for line in frustum_lines:
        path = trimesh.load_path([line])
        path.colors = np.asarray([frustum_color], dtype=np.uint8)
        scene.add_geometry(path)

    wire_color = np.array([255, 255, 255, 180], dtype=np.uint8)
    wireframe = create_oriented_wireframe(
        center=instance_center,
        extent=instance_extent,
        rotation=instance_axes,
        color_rgba=wire_color,
        thickness_ratio=0.003,
        min_thickness=0.002,
    )
    scene.add_geometry(wireframe)

    color_cluster_path = cluster_dir / "color_cluster.ply"
    if color_cluster_path.exists():
        cluster_geom = trimesh.load(str(color_cluster_path))
        if scene_rotation is not None or floor_z is not None:
            transform = np.eye(4, dtype=np.float32)
            if scene_rotation is not None:
                transform[:3, :3] = scene_rotation
            if floor_z is not None:
                transform[:3, 3] = [0.0, 0.0, -float(floor_z)]
            if isinstance(cluster_geom, trimesh.Scene):
                for geom in cluster_geom.geometry.values():
                    geom.apply_transform(transform)
            else:
                cluster_geom.apply_transform(transform)
        if isinstance(cluster_geom, trimesh.Scene):
            for geom in cluster_geom.geometry.values():
                scene.add_geometry(geom)
        else:
            scene.add_geometry(cluster_geom)
    else:
        print(f"[Heatmap] Warning: {color_cluster_path} not found, skip cluster geometry.")

    heatmap_scene = scene
    return heatmap_scene, heatmap_sphere_points, heatmap_values, heatmap_scene_diag, min_visible_train_dist


def _generate_views_for_instance(
    *,
    args: argparse.Namespace,
    instance_center: np.ndarray,
    instance_extent: np.ndarray,
    instance_axes: np.ndarray,
    heatmap_sphere_points: np.ndarray,
    heatmap_values: np.ndarray,
    scene_obb: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    scene_obb_margin: float,
    min_visible_train_dist: Optional[float],
) -> List[Dict]:
    views_aligned: List[Dict] = []

    if args.candidate_mode in ["geometric", "all"]:
        geom_views = generate_adaptive_views(
            obb_center=instance_center,
            obb_extent=instance_extent,
            obb_rotation=instance_axes,
            fov=args.fov,
            resolution=args.render_res,
            scene_obb=scene_obb,
            scene_obb_margin=scene_obb_margin,
            min_camera_distance=min_visible_train_dist,
        )
        for v in geom_views:
            v["name"] = "geom_" + v["name"]
        views_aligned.extend(geom_views)

    if args.candidate_mode in ["heatmap", "all"]:
        if heatmap_sphere_points.shape[0] > 0:
            heat_views = generate_heatmap_based_views(
                sphere_points=heatmap_sphere_points,
                heat_values=heatmap_values,
                obb_center=instance_center,
                obb_extent=instance_extent,
                obb_rotation=instance_axes,
                num_views=args.num_heatmap_views,
                fov=args.fov,
                resolution=args.render_res,
                scene_obb=scene_obb,
                scene_obb_margin=scene_obb_margin,
                min_camera_distance=min_visible_train_dist,
                min_angle_dist_deg=args.min_angle_dist_deg,
            )
            views_aligned.extend(heat_views)
        else:
            print("[Candidates] Warning: Heatmap data empty. Skipping heatmap views.")
            if args.candidate_mode == "heatmap":
                print("[Candidates] Fallback to geometric mode.")
                views_aligned = generate_adaptive_views(
                    obb_center=instance_center,
                    obb_extent=instance_extent,
                    obb_rotation=instance_axes,
                    fov=args.fov,
                    resolution=args.render_res,
                    scene_obb=scene_obb,
                    scene_obb_margin=scene_obb_margin,
                    min_camera_distance=min_visible_train_dist,
                )
                for v in views_aligned:
                    v["name"] = "geom_" + v["name"]

    return views_aligned


def _add_candidate_frustums_to_scene(
    *,
    heatmap_scene: Optional[trimesh.Scene],
    views_aligned: List[Dict],
    instance_extent: np.ndarray,
    heatmap_scene_diag: float,
) -> None:
    if heatmap_scene is None or not views_aligned:
        return

    candidate_frustum_depth = (
        heatmap_scene_diag * 0.01
        if heatmap_scene_diag > 1e-6
        else float(np.max(instance_extent) * 0.12)
    )
    cand_color = np.array([0, 120, 255, 220], dtype=np.uint8)
    for v in views_aligned:
        width = int(v["resolution"])
        height = int(v["resolution"])
        K = intrinsics_from_fov(width=width, height=height, fov_deg=float(v["fov"]))
        frustum_lines = build_camera_frustum_lines(
            camtoworld=v["c2w"],
            K=K,
            width=width,
            height=height,
            depth=candidate_frustum_depth,
            scene_rotation=None,
            floor_z=None,
        )
        for line in frustum_lines:
            path = trimesh.load_path([line])
            path.colors = np.asarray([cand_color], dtype=np.uint8)
            heatmap_scene.add_geometry(path)


def _save_candidate_views(
    *,
    save_dir: Path,
    views_aligned: List[Dict],
    scene_rotation: Optional[np.ndarray],
    floor_z: Optional[float],
) -> List[Dict]:
    T_o_a = _alignment_inverse_transform(scene_rotation, floor_z)
    candidate_views: List[Dict] = []
    for view in views_aligned:
        c2w_orig = (T_o_a @ view["c2w"]).astype(np.float32)
        candidate_views.append({**view, "c2w": c2w_orig})

    meta_out = []
    for v in candidate_views:
        entry = {
            "name": v["name"],
            "c2w": v["c2w"].tolist(),
            "fov": v["fov"],
            "resolution": v["resolution"],
        }
        if "occlusion_ratio" in v:
            entry["occlusion_ratio"] = float(v["occlusion_ratio"])
        if "visibility_ratio" in v:
            entry["visibility_ratio"] = float(v["visibility_ratio"])
        if "heatmap_score" in v:
            entry["heatmap_score"] = float(v["heatmap_score"])
        if "composite_score" in v:
            entry["composite_score"] = float(v["composite_score"])
        meta_out.append(entry)
    with open(save_dir / "cameras.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    return candidate_views


def _render_candidate_pngs(
    *,
    args: argparse.Namespace,
    data_dir: Path,
    instance_ids: List[int],
    save_dir: Path,
    candidate_views: List[Dict],
    gaussians_all: Optional[Tuple[torch.Tensor, ...]],
    render_device: Optional[torch.device],
    full_gaussian_count: Optional[int],
) -> None:
    if gaussians_all is None or render_device is None or full_gaussian_count is None:
        return

    instance_gaussian_ids = load_instance_gaussian_ids_multi(
        data_dir=data_dir, instance_ids=[int(i) for i in instance_ids]
    )
    if instance_gaussian_ids is None:
        print("[Mask] Warning: total_point_ids_list missing; RGBA masks will be skipped.")
    elif instance_gaussian_ids.size == 0:
        print("[Mask] Warning: empty instance indices; RGBA masks will be skipped.")
    elif (
        int(instance_gaussian_ids.min(initial=0)) < 0
        or int(instance_gaussian_ids.max(initial=-1)) >= int(gaussians_all[0].shape[0])
    ):
        print("[Mask] Warning: instance indices out of range; RGBA masks will be skipped.")
        instance_gaussian_ids = None

    gaussians = gaussians_all
    if args.render_instance_only:
        if instance_gaussian_ids is None:
            print("[Render] Warning: total_point_ids_list missing in tracking; rendering all gaussians.")
        else:
            try:
                gaussians = filter_gaussians_by_ids(gaussians, instance_gaussian_ids)
                print(
                    f"[Render] Rendering instance-only gaussians: {len(instance_gaussian_ids)} / {full_gaussian_count}"
                )
            except Exception as exc:
                print(f"[Render] Warning: failed to filter instance gaussians ({exc}); rendering all gaussians.")
    else:
        print(f"[Render] Rendering all gaussians: {full_gaussian_count}")

    flip_lr = bool(args.render_flip_lr)
    mask_topk = max(0, int(getattr(args, "export_topk", 1)))
    depths_dir = save_dir / "depths"
    depths_vis_dir = save_dir / "depths_vis"
    if mask_topk > 0:
        depths_dir.mkdir(parents=True, exist_ok=True)
        depths_vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Render] Rendering {len(candidate_views)} views...")
    for view_idx, v in enumerate(candidate_views):
        img = render_camera_view(
            gaussians=gaussians,
            c2w_np=v["c2w"],
            width=v["resolution"],
            height=v["resolution"],
            fov_deg=v["fov"],
            device=render_device,
        )
        if flip_lr:
            img = np.fliplr(img)
        out_path = save_dir / f"{v['name']}.png"
        imageio.imwrite(out_path, img)
        print(f"  -> Saved {out_path.name}")

        if view_idx < mask_topk:
            width = int(v["resolution"])
            height = int(v["resolution"])
            K_np = intrinsics_from_fov(width=width, height=height, fov_deg=float(v["fov"]))
            K = torch.from_numpy(K_np).to(device=render_device, dtype=torch.float32).unsqueeze(0)
            depth = render_depth(
                gaussians=gaussians,
                c2w_np=v["c2w"],
                K=K,
                width=width,
                height=height,
                device=render_device,
            )
            depth_np = depth.detach().float().cpu().numpy().astype(np.float32, copy=False)
            if flip_lr:
                depth_np = np.fliplr(depth_np)
            np.save(depths_dir / f"{v['name']}.npy", depth_np)
            depth_vis = _depth_to_vis_rgb(depth_np)
            imageio.imwrite(depths_vis_dir / f"{v['name']}.png", depth_vis)

        if instance_gaussian_ids is not None and view_idx < mask_topk:
            alpha_bin = render_visible_instance_mask(
                gaussians_all=gaussians_all,
                instance_indices=instance_gaussian_ids,
                c2w_np=v["c2w"],
                width=v["resolution"],
                height=v["resolution"],
                fov_deg=v["fov"],
                device=render_device,
                alpha_threshold=float(args.occlusion_alpha_threshold),
                white_threshold=float(args.occlusion_white_threshold),
            )
            if flip_lr:
                alpha_bin = np.fliplr(alpha_bin)
            rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = img
            rgba[..., 3] = alpha_bin
            mask_out_path = save_dir / f"{v['name']}_mask.png"
            imageio.imwrite(mask_out_path, rgba)


def rank_views_by_strategy_two(
    views: List[Dict],
    alpha: float = 2.0,
    beta: float = 0.5,
) -> List[Dict]:
    """
    Weighted Product Strategy (strategy two):
        Score = H * (1 - O)^alpha * V^beta

    Where:
        H: heatmap_score (higher is better)
        O: occlusion_ratio (lower is better)
        V: visibility_ratio (higher is better)
    """
    if not views:
        return []

    print(f"[Ranking] Scoring {len(views)} views using weighted product strategy...")
    for view in views:
        heat = float(view.get("heatmap_score", 0.0))
        occlusion = float(view.get("occlusion_ratio", 1.0))
        visibility = float(view.get("visibility_ratio", 0.0))

        heat = max(0.0, heat)
        occlusion = float(np.clip(occlusion, 0.0, 1.0))
        visibility = float(np.clip(visibility, 0.0, 1.0))

        score = heat * ((1.0 - occlusion) ** float(alpha)) * (visibility ** float(beta))
        view["composite_score"] = float(score)

    ranked = sorted(views, key=lambda v: float(v.get("composite_score", 0.0)), reverse=True)
    for i, v in enumerate(ranked[:5]):
        print(
            f"  Rank {i+1}: {v.get('name', '<unnamed>')} | Score={v.get('composite_score', 0.0):.4f} "
            f"(H={v.get('heatmap_score', 0.0):.2f}, O={v.get('occlusion_ratio', 1.0):.2f}, V={v.get('visibility_ratio', 0.0):.2f})"
        )
    return ranked


def _export_rank01_assets(
    *,
    out_root: Path,
    instance_id: int,
    save_dir: Path,
    views_aligned: List[Dict],
    export_topk: int = 1,
    render_flip_lr: bool = False,
) -> None:
    """
    Copy ranked view renders (+ masks when available) to centralized folders:
      - {out_root}/images/instance_{id}_rank_XX.png
      - {out_root}/projected_mask/instance_{id}_rank_XX_mask.png
    """
    if not views_aligned or int(export_topk) <= 0:
        return

    images_dir = out_root / "images"
    masks_dir = out_root / "projected_mask"
    depths_dir = out_root / "depths"
    depths_vis_dir = out_root / "depths_vis"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    depths_dir.mkdir(parents=True, exist_ok=True)
    depths_vis_dir.mkdir(parents=True, exist_ok=True)

    cameras_json_path = images_dir / "cameras.json"
    cameras_by_name: Dict[str, Dict] = {}
    if cameras_json_path.exists():
        try:
            with open(cameras_json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                for e in existing:
                    if isinstance(e, dict) and isinstance(e.get("file_name"), str):
                        cameras_by_name[e["file_name"]] = e
        except Exception as exc:
            print(f"[Export] Warning: failed to read existing cameras.json ({exc}); rebuilding it.")
            cameras_by_name = {}

    topk = min(int(export_topk), len(views_aligned))
    for rank_idx in range(topk):
        view_name = str(views_aligned[rank_idx].get("name", ""))
        if not view_name:
            continue
        rank_str = f"{rank_idx + 1:02d}"

        src_img = save_dir / f"{view_name}.png"
        dst_img = images_dir / f"instance_{int(instance_id)}_rank_{rank_str}.png"
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
            try:
                width = int(views_aligned[rank_idx].get("resolution", 0))
                height = int(views_aligned[rank_idx].get("resolution", 0))
                fov = float(views_aligned[rank_idx].get("fov", 0.0))
                K = intrinsics_from_fov(width=width, height=height, fov_deg=fov)
                c2w = np.asarray(views_aligned[rank_idx].get("c2w"), dtype=np.float32)
                if c2w.shape == (3, 4):
                    c2w = np.vstack([c2w, np.array([0, 0, 0, 1], dtype=np.float32)])
                w2c = np.linalg.inv(c2w).astype(np.float32)
                cameras_by_name[dst_img.name] = {
                    "file_name": dst_img.name,
                    "instance_id": int(instance_id),
                    "rank": int(rank_idx + 1),
                    "render_flip_lr": bool(render_flip_lr),
                    "coordinate_system": "checkpoint_world_opencv",
                    "source_view_name": view_name,
                    "width": int(width),
                    "height": int(height),
                    "fov": float(fov),
                    "K": K.tolist(),
                    "c2w": c2w.tolist(),
                    "w2c": w2c.tolist(),
                }
            except Exception as exc:
                print(f"[Export] Warning: failed to write camera entry for {dst_img.name} ({exc})")
        else:
            print(f"[Export] Warning: view image not found: {src_img}")

        src_mask = save_dir / f"{view_name}_mask.png"
        dst_mask = masks_dir / f"instance_{int(instance_id)}_rank_{rank_str}_mask.png"
        if src_mask.exists():
            shutil.copy2(src_mask, dst_mask)
        else:
            print(f"[Export] Warning: view mask not found: {src_mask}")

        src_depth = save_dir / "depths" / f"{view_name}.npy"
        dst_depth = depths_dir / f"instance_{int(instance_id)}_rank_{rank_str}.npy"
        if src_depth.exists():
            shutil.copy2(src_depth, dst_depth)
        else:
            print(f"[Export] Warning: view depth not found: {src_depth}")

        src_depth_vis = save_dir / "depths_vis" / f"{view_name}.png"
        dst_depth_vis = depths_vis_dir / f"instance_{int(instance_id)}_rank_{rank_str}.png"
        if src_depth_vis.exists():
            shutil.copy2(src_depth_vis, dst_depth_vis)
        else:
            print(f"[Export] Warning: view depth vis not found: {src_depth_vis}")

    # Save/update cameras.json in images/ directory.
    try:
        cameras_out = [cameras_by_name[k] for k in sorted(cameras_by_name.keys())]
        with open(cameras_json_path, "w", encoding="utf-8") as f:
            json.dump(cameras_out, f, indent=2)
    except Exception as exc:
        print(f"[Export] Warning: failed to save cameras.json ({exc})")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    cluster_dir = data_dir / "cluster_result"
    out_root = cluster_dir / "candidate_views"
    debug_root = out_root / "debug"
    out_root.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)
    atomic_path = cluster_dir / "atomic_geometry.pt"
    if not atomic_path.exists():
        raise FileNotFoundError(f"Missing atomic geometry: {atomic_path}")

    atomic = torch.load(atomic_path, map_location="cpu", weights_only=False)
    atomic_instance_ids = sorted(int(k) for k in atomic.keys())
    merge_groups, merge_mapping = build_merge_groups(atomic_instance_ids, MERGE_PAIRS if args.legacy_scene_rules else [])

    if args.instance_id is None:
        requested_ids = atomic_instance_ids
    else:
        requested_id = int(args.instance_id)
        if requested_id not in atomic:
            raise KeyError(f"Instance {requested_id} not found in {atomic_path}")
        rep = int(merge_mapping.get(requested_id, requested_id))
        if rep != requested_id:
            print(f"[Instance {requested_id}] Merged into {rep}; processing representative.")
        requested_ids = [rep]

    requested_ids = [int(merge_mapping.get(i, i)) for i in requested_ids]
    before = len(requested_ids)
    ignored_ids = HARDCODED_IGNORED_IDS if args.legacy_scene_rules else set()
    instance_ids = sorted({i for i in requested_ids if i not in ignored_ids})
    skipped = before - len(instance_ids)
    if skipped > 0:
        print(f"[Instances] Skipping {skipped} instances (hardcoded ignore list).")
    if not instance_ids:
        print("[Instances] No instances left to process after applying merge/ignore rules.")
        return

    view_data = _load_view_data(data_dir)
    _, gaussians_all, render_device, full_gaussian_count = _prepare_gaussians_for_rendering(
        args, data_dir
    )
    scene_obb, scene_obb_margin = _prepare_scene_obb(args, atomic)

    for instance_id in instance_ids:
        print(f"\n[Instance {instance_id}] Processing...")
        member_ids = merge_groups.get(int(instance_id), [int(instance_id)])
        instance_info = atomic[int(instance_id)]
        instance_center, instance_extent, instance_axes = merge_instance_obbs(atomic, member_ids)
        scene_rotation = None
        floor_z = None
        if "scene_R" in instance_info:
            scene_rotation = np.asarray(instance_info["scene_R"], dtype=np.float32)
        if "scene_floor_z" in instance_info:
            floor_z = float(instance_info["scene_floor_z"])

        (
            heatmap_scene,
            heatmap_sphere_points,
            heatmap_values,
            heatmap_scene_diag,
            min_visible_train_dist,
        ) = _maybe_build_heatmap_scene(
            args=args,
            cluster_dir=cluster_dir,
            view_data=view_data,
            instance_ids=member_ids,
            instance_center=instance_center,
            instance_extent=instance_extent,
            instance_axes=instance_axes,
            scene_rotation=scene_rotation,
            floor_z=floor_z,
        )

        # -------------------------------------------------------
        # 2) Candidate view generation (+ optional rendering)
        # -------------------------------------------------------
        if args.gen_candidates:
            save_dir = debug_root / f"instance_{instance_id}_candidate_views"
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Candidates] Generating views -> {save_dir}")
            print(f"[Candidates] Mode: {args.candidate_mode}")

            views_aligned = _generate_views_for_instance(
                args=args,
                instance_center=instance_center,
                instance_extent=instance_extent,
                instance_axes=instance_axes,
                heatmap_sphere_points=heatmap_sphere_points,
                heatmap_values=heatmap_values,
                scene_obb=scene_obb,
                scene_obb_margin=scene_obb_margin,
                min_visible_train_dist=min_visible_train_dist,
            )

            if (
                args.check_occlusion
                and HAS_GSPLAT
                and rasterization is not None
                and gaussians_all is not None
                and render_device is not None
            ):
                instance_indices = load_instance_gaussian_ids_multi(data_dir, member_ids)
                if instance_indices is None:
                    print("[Occlusion] Warning: total_point_ids_list missing; skip occlusion check.")
                elif instance_indices.size == 0:
                    print("[Occlusion] Warning: empty instance indices; skip occlusion check.")
                else:
                    print("[Occlusion] Starting occlusion check (mask-based)...")
                    T_o_a = _alignment_inverse_transform(scene_rotation, floor_z)
                    valid_views: List[Dict] = []
                    dropped: List[Tuple[float, Dict]] = []
                    for v in views_aligned:
                        view_for_render = dict(v)
                        view_for_render["c2w"] = (T_o_a @ v["c2w"]).astype(np.float32)
                        occ_ratio = calculate_occlusion_by_mask(
                            view=view_for_render,
                            gaussians_all=gaussians_all,
                            instance_indices=instance_indices,
                            device=render_device,
                            alpha_threshold=args.occlusion_alpha_threshold,
                            white_threshold=args.occlusion_white_threshold,
                        )
                        v["occlusion_ratio"] = float(occ_ratio)
                        v["name"] = f"{v['name']}_occ{occ_ratio:.2f}"
                        if occ_ratio <= float(args.max_occlusion_ratio):
                            valid_views.append(v)
                            print(f"  -> View {v['name']}: Occlusion={occ_ratio:.2%}")
                        else:
                            dropped.append((float(occ_ratio), v))
                            print(f"  -> [DROPPED] View {v['name']}: Occlusion={occ_ratio:.2%}")

                    if not valid_views and dropped:
                        dropped.sort(key=lambda x: x[0])
                        best = dropped[0][1]
                        print(f"[Occlusion] Warning: all views dropped; keeping best one: {best['name']}")
                        valid_views = [best]
                    views_aligned = valid_views

            if (
                args.check_self_occlusion
                and HAS_GSPLAT
                and rasterization is not None
                and gaussians_all is not None
                and render_device is not None
                and views_aligned
            ):
                instance_indices = load_instance_gaussian_ids_multi(data_dir, member_ids)
                if instance_indices is None:
                    print("[Visibility] Warning: total_point_ids_list missing; skip self-occlusion check.")
                elif instance_indices.size == 0:
                    print("[Visibility] Warning: empty instance indices; skip self-occlusion check.")
                else:
                    print("[Visibility] Starting point visibility check (self-occlusion)...")
                    gaussians_inst = filter_gaussians_by_ids(gaussians_all, instance_indices)
                    T_o_a = _alignment_inverse_transform(scene_rotation, floor_z)
                    for v in views_aligned:
                        view_for_render = dict(v)
                        view_for_render["c2w"] = (T_o_a @ v["c2w"]).astype(np.float32)
                        vis_ratio = calculate_point_visibility_ratio(
                            view=view_for_render,
                            gaussians_inst=gaussians_inst,
                            device=render_device,
                            obb_extent=instance_extent,
                            scale_ratio=0.02,
                        )
                        v["visibility_ratio"] = float(vis_ratio)
                        v["name"] = f"{v['name']}_vis{vis_ratio:.2f}"
                        print(f"  -> View {v['name']}: Visibility={vis_ratio:.2%}")

            if views_aligned:
                views_aligned = rank_views_by_strategy_two(
                    views_aligned,
                    alpha=2.0,
                    beta=0.5,
                )
                pad = max(2, len(str(len(views_aligned))))
                for rank_idx, v in enumerate(views_aligned, start=1):
                    name = str(v.get("name", f"view_{rank_idx}"))
                    if not name.startswith("rank"):
                        name = f"rank{rank_idx:0{pad}d}_{name}"
                    name = re.sub(
                        r"(^|_)heat_\d+_h([0-9]+(?:\.[0-9]+)?)",
                        r"\1heat\2",
                        name,
                    )
                    name = re.sub(r"__+", "_", name).strip("_")
                    v["name"] = name

            _add_candidate_frustums_to_scene(
                heatmap_scene=heatmap_scene,
                views_aligned=views_aligned,
                instance_extent=instance_extent,
                heatmap_scene_diag=heatmap_scene_diag,
            )

            candidate_views = _save_candidate_views(
                save_dir=save_dir,
                views_aligned=views_aligned,
                scene_rotation=scene_rotation,
                floor_z=floor_z,
            )
            _render_candidate_pngs(
                args=args,
                data_dir=data_dir,
                instance_ids=member_ids,
                save_dir=save_dir,
                candidate_views=candidate_views,
                gaussians_all=gaussians_all,
                render_device=render_device,
                full_gaussian_count=full_gaussian_count,
            )
            _export_rank01_assets(
                out_root=out_root,
                instance_id=instance_id,
                save_dir=save_dir,
                views_aligned=candidate_views,
                export_topk=int(args.export_topk),
                render_flip_lr=bool(args.render_flip_lr),
            )

        if heatmap_scene is not None:
            heatmap_out_path = debug_root / f"instance_{instance_id}_heatmap.glb"
            heatmap_scene.export(str(heatmap_out_path))
            print(f"[Heatmap] Saved {heatmap_out_path}")


if __name__ == "__main__":
    main()
