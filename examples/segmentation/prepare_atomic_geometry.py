import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import open3d as o3d
import torch
import trimesh
import trimesh.transformations as tf
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streamlined: Instance Geometry with Gravity Alignment & Wireframe Viz."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root dir.")
    parser.add_argument("--ckpt", type=str, required=True, help="3DGS checkpoint (.pt).")
    parser.add_argument("--cluster_dir", type=str, default="cluster_result")
    parser.add_argument("--labels_name", type=str, default="instance_labels.npy")
    parser.add_argument("--output_name", type=str, default="atomic_geometry.pt")
    parser.add_argument("--viz_name", type=str, default="scene_viz.glb")

    # 几何处理参数
    parser.add_argument("--min_points", type=int, default=50)
    parser.add_argument("--outlier_nb_neighbors", type=int, default=50)
    parser.add_argument("--outlier_std_ratio", type=float, default=2.0)

    # 可视化参数 (线框相关参数已恢复)
    parser.add_argument("--wire_thickness_ratio", type=float, default=0.003)
    parser.add_argument("--wire_min_thickness", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)

    # 重力对齐参数
    parser.add_argument("--skip_align", action="store_true", help="Skip gravity alignment.")
    parser.add_argument("--plane_distance_threshold", type=float, default=0.05)
    parser.add_argument("--plane_num_iterations", type=int, default=2000)
    parser.add_argument("--plane_ransac_n", type=int, default=5)
    parser.add_argument("--align_translate_floor", action="store_true", help="Translate floor to Z=0.")
    parser.add_argument(
        "--floor_idx",
        type=int,
        default=-1,
        help="Index of candidate plane to use (0-4); -1 means auto pick best.",
    )

    # 包围盒策略参数
    parser.add_argument(
        "--pca_ratio_threshold",
        type=float,
        default=0.7,
        help="Switch to full 6-DoF PCA OBB if volume is < threshold x upright OBB.",
    )

    return parser.parse_args()


def load_gaussians_means(ckpt_path: str, device: torch.device) -> torch.Tensor:
    ckpt_path = os.path.expanduser(ckpt_path)
    print(f"[Loader] Loading 3DGS checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    return payload["splats"]["means"]


def _rotation_matrix_from_vectors(vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
    """Compute rotation matrix that rotates vec_from to vec_to."""
    vec_from = vec_from / (np.linalg.norm(vec_from) + 1e-12)
    vec_to = vec_to / (np.linalg.norm(vec_to) + 1e-12)
    v = np.cross(vec_from, vec_to)
    c = np.dot(vec_from, vec_to)
    s = np.linalg.norm(v)

    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
    return r


def find_planes_from_top_instances(
    points_np: np.ndarray,
    instance_labels: np.ndarray,
    top_k: int = 10,
    dist_thresh: float = 0.05,
    ransac_n: int = 3,
    iters: int = 1000
) -> List[Dict]:
    """Find dominant planes within the largest instances."""
    print(f"[Align] Finding planes in top {top_k} largest instances...")

    unique_ids, counts = np.unique(instance_labels, return_counts=True)
    valid_mask = unique_ids >= 0
    unique_ids, counts = unique_ids[valid_mask], counts[valid_mask]

    if len(unique_ids) == 0:
        return []

    top_indices = np.argsort(-counts)[:min(len(counts), top_k)]
    top_ids = unique_ids[top_indices]

    candidates = []
    for i, inst_id in enumerate(top_ids):
        mask = (instance_labels == inst_id)
        inst_pts = points_np[mask]

        if len(inst_pts) < ransac_n:
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inst_pts)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=ransac_n, num_iterations=iters
        )

        if len(inliers) == 0:
            continue

        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        normal /= np.linalg.norm(normal) + 1e-12
        inlier_pts = inst_pts[inliers]
        center = np.mean(inlier_pts, axis=0)

        candidates.append({
            "instance_id": int(inst_id),
            "model": plane_model,
            "normal": normal,
            "inlier_count": len(inliers),
            "center": center,
            "total_points": len(inst_pts)
        })

    candidates.sort(key=lambda x: x['inlier_count'], reverse=True)
    return candidates[:5]


def align_to_specific_plane(points_np: np.ndarray, plane_info: Dict, translate_floor: bool) -> Dict:
    """Align scene so plane normal points to +Z."""
    pts = points_np.astype(np.float32)
    normal = plane_info["normal"]
    center = plane_info["center"]

    centroid = np.mean(pts, axis=0)
    if np.dot(normal, centroid - center) < 0:
        normal = -normal
        print("[Align] Flipping normal to point towards centroid.")

    target_axis = np.array([0, 0, 1], dtype=np.float64)
    R = _rotation_matrix_from_vectors(normal, target_axis).astype(np.float32)

    aligned_pts = (pts @ R.T)
    floor_z = 0.0

    if translate_floor:
        rotated_center = (center @ R.T)
        floor_z = float(rotated_center[2])
        aligned_pts[:, 2] -= floor_z
        print(f"[Align] Translated floor from Z={floor_z:.2f} to Z=0.0")

    return {"points": aligned_pts, "R": R, "floor_z": floor_z}


def _min_area_rect_2d(points: np.ndarray) -> Dict[str, Any]:
    """Compute 2D minimum-area bounding rectangle (simplified)."""
    if len(points) < 3:
         min_xy, max_xy = points.min(axis=0), points.max(axis=0)
         return {"center": (min_xy+max_xy)/2, "extents": max_xy-min_xy, "yaw": 0.0}

    # Using standard scipy ConvexHull if available, else skipping hull optimization
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except ImportError:
        hull_points = points

    best = {"area": np.inf, "yaw": 0.0, "min_xy": None, "max_xy": None}

    edges = hull_points[1:] - hull_points[:-1]
    edges = np.vstack([edges, hull_points[0] - hull_points[-1]])
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.unique(np.mod(angles, np.pi/2))

    for theta in angles:
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s], [-s, c]])
        proj = hull_points @ R.T
        min_xy, max_xy = proj.min(axis=0), proj.max(axis=0)
        area = (max_xy[0]-min_xy[0]) * (max_xy[1]-min_xy[1])
        if area < best["area"]:
            best = {"area": area, "yaw": theta, "min_xy": min_xy, "max_xy": max_xy}

    yaw = best["yaw"]
    c, s = np.cos(yaw), np.sin(yaw)
    R_inv = np.array([[c, -s], [s, c]])
    center_local = (best["min_xy"] + best["max_xy"]) / 2
    center_world = R_inv @ center_local

    return {"center": center_world, "extents": best["max_xy"]-best["min_xy"], "yaw": yaw}


def process_single_instance(
    instance_id: int,
    points_np: np.ndarray,
    nb_neighbors: int,
    std_ratio: float,
    pca_ratio_threshold: float = 0.7,
) -> Dict:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np))

    if len(points_np) >= max(10, nb_neighbors):
        _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        clean_pcd = pcd.select_by_index(inlier_idx) if len(inlier_idx) > 10 else pcd
    else:
        clean_pcd = pcd

    clean_pts = np.asarray(clean_pcd.points, dtype=np.float32)

    if len(clean_pts) < 4:
        obb_center = np.mean(clean_pts, axis=0) if len(clean_pts) > 0 else np.zeros(3, dtype=np.float32)
        obb_extent = np.zeros(3, dtype=np.float32)
        obb_rot = np.eye(3, dtype=np.float32)
        return {
            "geometry_points": clean_pts,
            "obb_center": obb_center,
            "obb_extent": obb_extent,
            "obb_rotation": obb_rot,
            "aabb_min": clean_pcd.get_min_bound(),
            "aabb_max": clean_pcd.get_max_bound(),
        }

    # -----------------------------------------------------------------
    # A) Upright OBB (Z-up, yaw only)
    # -----------------------------------------------------------------
    rect = _min_area_rect_2d(clean_pts[:, :2])
    z_min, z_max = clean_pts[:, 2].min(), clean_pts[:, 2].max()

    up_center = np.array(
        [rect["center"][0], rect["center"][1], (z_min + z_max) / 2], dtype=np.float32
    )
    up_extent = np.array(
        [rect["extents"][0], rect["extents"][1], z_max - z_min], dtype=np.float32
    )
    up_extent = np.maximum(up_extent, 1e-6)

    yaw = float(rect["yaw"])
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    up_rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    vol_upright = float(np.prod(up_extent))

    # -----------------------------------------------------------------
    # B) PCA OBB (6-DoF)
    # -----------------------------------------------------------------
    pca_center = None
    pca_extent = None
    pca_rot = None
    vol_pca = float("inf")
    try:
        obb_pca_obj = clean_pcd.get_oriented_bounding_box()
        pca_center = np.asarray(obb_pca_obj.center, dtype=np.float32)
        pca_extent = np.asarray(obb_pca_obj.extent, dtype=np.float32)
        pca_rot = np.asarray(obb_pca_obj.R, dtype=np.float32)
        vol_pca = float(np.prod(np.maximum(pca_extent, 1e-6)))
    except Exception:
        pass

    # -----------------------------------------------------------------
    # C) Decision: use PCA only when it is significantly tighter
    # -----------------------------------------------------------------
    ratio = vol_pca / (vol_upright + 1e-9)
    if ratio < float(pca_ratio_threshold) and pca_center is not None:
        final_center, final_extent, final_rotation = pca_center, pca_extent, pca_rot
    else:
        final_center, final_extent, final_rotation = up_center, up_extent, up_rot

    return {
        "geometry_points": clean_pts,
        "obb_center": final_center,
        "obb_extent": final_extent,
        "obb_rotation": final_rotation,
        "aabb_min": clean_pcd.get_min_bound(),
        "aabb_max": clean_pcd.get_max_bound()
    }


# === 恢复的线框可视化逻辑 ===
def create_oriented_wireframe(center, extent, rotation, color_rgba, thickness_ratio, min_thickness) -> trimesh.Trimesh:
    """Builds an OBB wireframe using 12 thin box meshes."""
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

    # X-axis edges
    for dy in (-hy, hy):
        for dz in (-hz, hz):
            add_stick([extent[0], t, t], [0, dy, dz])

    # Y-axis edges
    for dx in (-hx, hx):
        for dz in (-hz, hz):
            add_stick([t, extent[1], t], [dx, 0, dz])

    # Z-axis edges
    for dx in (-hx, hx):
        for dy in (-hy, hy):
            add_stick([t, t, extent[2]], [dx, dy, 0])

    wireframe = trimesh.util.concatenate(sticks)
    wireframe.visual.face_colors = color_rgba

    # Apply OBB transform
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    wireframe.apply_transform(transform)

    return wireframe


def save_viz_glb(geometries: Dict, path: Path, seed: int, thickness_ratio: float, min_thickness: float):
    scene = trimesh.Scene()

    print(f"[Viz] Exporting {len(geometries)} instances to GLB...")
    for uid, data in geometries.items():
        # Color generation
        rng_inst = np.random.default_rng(seed + int(uid) * 1315423911)
        color_rgb = rng_inst.integers(50, 256, size=3, dtype=np.uint8)
        color_rgba = np.append(color_rgb, 255)

        # 1. Add Point Cloud
        pts = data["geometry_points"]

        # Create colors array matching points
        pcd_colors = np.tile(color_rgba, (len(pts), 1))
        pcd = trimesh.points.PointCloud(pts, colors=pcd_colors)
        scene.add_geometry(pcd)

        # 2. Add Wireframe (The restored logic)
        wireframe = create_oriented_wireframe(
            center=data["obb_center"],
            extent=data["obb_extent"],
            rotation=data["obb_rotation"],
            color_rgba=color_rgba,
            thickness_ratio=thickness_ratio,
            min_thickness=min_thickness
        )
        scene.add_geometry(wireframe)

    scene.export(str(path))
    print(f"[Viz] Saved to {path}")


def main():
    args = parse_args()
    device = torch.device("cpu")

    # 1. Load Data
    means_np = load_gaussians_means(args.ckpt, device).cpu().numpy()

    # === [新增] 自适应尺度计算 ===
    # 计算整个场景的包围盒对角线长度
    scene_min = means_np.min(axis=0)
    scene_max = means_np.max(axis=0)
    scene_extent = scene_max - scene_min
    scene_diag = float(np.linalg.norm(scene_extent))
    print(f"[Auto-Scale] Scene Diagonal: {scene_diag:.4f} units")

    # 1. 自适应平面阈值：设为场景尺度的 0.5% - 1%
    # 如果场景是 10米大，阈值就是 5cm-10cm。
    # 如果场景是 1.0单位大，阈值就是 0.005-0.01。
    plane_thresh_flag = "--plane_distance_threshold"
    plane_thresh_user_set = any(
        arg == plane_thresh_flag or arg.startswith(f"{plane_thresh_flag}=") for arg in sys.argv[1:]
    )
    if (not plane_thresh_user_set) and args.plane_distance_threshold == 0.05:
        adaptive_plane_thresh = scene_diag * 0.005
        print(
            f"[Auto-Scale] Overriding plane_distance_threshold: {args.plane_distance_threshold} -> {adaptive_plane_thresh:.6f}"
        )
        args.plane_distance_threshold = adaptive_plane_thresh

    # 2. 自适应线框最小厚度：设为场景尺度的 0.02%
    wire_min_flag = "--wire_min_thickness"
    wire_min_user_set = any(
        arg == wire_min_flag or arg.startswith(f"{wire_min_flag}=") for arg in sys.argv[1:]
    )
    if (not wire_min_user_set) and args.wire_min_thickness == 0.002:
        adaptive_wire_thick = scene_diag * 0.0005
        print(
            f"[Auto-Scale] Overriding wire_min_thickness: {args.wire_min_thickness} -> {adaptive_wire_thick:.6f}"
        )
        args.wire_min_thickness = adaptive_wire_thick

    # 3. (可选) 检查 epsilon 风险
    if scene_diag < 1e-4:
        print(
            "[Auto-Scale] WARNING: Scene scale is extremely small (< 1e-4). Hardcoded epsilons (1e-6) might cause issues."
        )
    # ===============================

    lbl_path = Path(args.data_dir) / args.cluster_dir / args.labels_name
    print(f"[Loader] Loading labels: {lbl_path}")
    labels = np.load(lbl_path).reshape(-1)

    if len(labels) != len(means_np):
        raise ValueError(f"Shape mismatch: Means {len(means_np)} vs Labels {len(labels)}")

    total_points_input = int(len(means_np))
    total_points_labeled = int(np.sum(labels >= 0))
    print(
        f"[Stats] Input points: {total_points_input} (labeled instances: {total_points_labeled}, "
        f"background/unlabeled: {total_points_input - total_points_labeled})"
    )

    # 2. Gravity Alignment
    align_info = None
    if not args.skip_align:
        candidates = find_planes_from_top_instances(
            means_np, labels,
            top_k=10,
            dist_thresh=args.plane_distance_threshold,
            ransac_n=args.plane_ransac_n,
            iters=args.plane_num_iterations
        )

        selected_plane = None
        if candidates:
            print("\n [Top Candidates]")
            print(f" {'Idx':<4} | {'InstID':<6} | {'Inliers':<8} | {'Normal Z'}")
            for idx, c in enumerate(candidates):
                print(f" {idx:<4} | {c['instance_id']:<6} | {c['inlier_count']:<8} | {c['normal'][2]:.3f}")
            print("-" * 30)

            choice = 0
            if args.floor_idx >= 0 and args.floor_idx < len(candidates):
                choice = args.floor_idx
                print(f"[Align] Manual selection: Index {choice}")
            else:
                print(f"[Align] Auto selection: Index 0")
            selected_plane = candidates[choice]

        if selected_plane:
            align_data = align_to_specific_plane(
                means_np, selected_plane, args.align_translate_floor
            )
            means_np = align_data["points"]
            align_info = {"R": align_data["R"], "floor_z": align_data["floor_z"]}
        else:
            print("[Align] WARN: No valid instance planes found. Skipping alignment.")

    # 3. Process Geometry
    unique_ids = np.unique(labels[labels >= 0])
    results = {}
    total_points_kept_pre_denoise = 0
    total_points_output = 0
    skipped_by_min_points = 0

    print(f"[Process] Computing geometry for {len(unique_ids)} instances...")
    for uid in tqdm(unique_ids):
        mask = (labels == uid)
        pts = means_np[mask]

        if len(pts) < args.min_points:
            skipped_by_min_points += 1
            continue
        total_points_kept_pre_denoise += int(len(pts))

        res = process_single_instance(
            uid,
            pts,
            args.outlier_nb_neighbors,
            args.outlier_std_ratio,
            pca_ratio_threshold=args.pca_ratio_threshold,
        )
        total_points_output += int(len(res["geometry_points"]))

        if align_info:
            res["scene_R"] = align_info["R"]
            res["scene_floor_z"] = align_info["floor_z"]

        results[int(uid)] = res

    print(
        f"[Stats] Output instances: {len(results)} (skipped by min_points: {skipped_by_min_points})"
    )
    print(
        f"[Stats] Output points: {total_points_output} (pre-denoise kept: {total_points_kept_pre_denoise})"
    )

    # 4. Save
    out_dir = Path(args.data_dir) / args.cluster_dir
    out_dir.mkdir(exist_ok=True, parents=True)

    torch.save(results, out_dir / args.output_name)
    save_viz_glb(
        results,
        out_dir / args.viz_name,
        args.seed,
        args.wire_thickness_ratio,
        args.wire_min_thickness
    )
    print("[Done]")

if __name__ == "__main__":
    main()
