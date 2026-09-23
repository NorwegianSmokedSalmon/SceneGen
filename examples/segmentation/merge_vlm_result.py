import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Merge VLM results and export masks/PLY.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Dataset root.")
    return parser.parse_args()


def read_mask(mask_path: str | Path) -> np.ndarray | None:
    mask_path = str(mask_path)
    if mask_path.endswith(".npy"):
        try:
            mask = np.load(mask_path)
        except Exception as exc:
            print(f"[Mask] Error loading npy {mask_path}: {exc}")
            return None
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32, copy=False)


def get_mask_properties(
    mask_bool: np.ndarray, *, border_margin_px: int = 20
) -> tuple[np.ndarray | None, bool]:
    if not np.any(mask_bool):
        return None, False

    rows, cols = np.where(mask_bool)
    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(cols.min()), int(cols.max())

    center_y = (y_min + y_max) / 2.0
    center_x = (x_min + x_max) / 2.0

    h, w = mask_bool.shape
    margin = int(border_margin_px)
    touching = (
        (x_min < margin)
        or (y_min < margin)
        or (x_max > (w - margin))
        or (y_max > (h - margin))
    )

    return np.array([center_x, center_y], dtype=np.float32), touching


def _generate_contrast_palette(num_entries: int, seed: int = 42) -> np.ndarray:
    if num_entries <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    palette = np.zeros((num_entries, 3), dtype=np.uint8)
    if num_entries == 1:
        return palette

    rng = np.random.default_rng(seed)
    n = num_entries - 1
    phi = 0.6180339887498949
    h0 = float(rng.random())
    hues = (h0 + phi * np.arange(n, dtype=np.float32)) % 1.0
    sats = 0.35 + 0.50 * rng.random(n, dtype=np.float32)
    vals = 0.65 + 0.33 * rng.random(n, dtype=np.float32)

    h6 = hues * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    p = vals * (1.0 - sats)
    q = vals * (1.0 - f * sats)
    t = vals * (1.0 - (1.0 - f) * sats)
    i_mod = i % 6

    r = np.empty(n, dtype=np.float32)
    g = np.empty(n, dtype=np.float32)
    b = np.empty(n, dtype=np.float32)

    m0 = i_mod == 0
    r[m0], g[m0], b[m0] = vals[m0], t[m0], p[m0]
    m1 = i_mod == 1
    r[m1], g[m1], b[m1] = q[m1], vals[m1], p[m1]
    m2 = i_mod == 2
    r[m2], g[m2], b[m2] = p[m2], vals[m2], t[m2]
    m3 = i_mod == 3
    r[m3], g[m3], b[m3] = p[m3], q[m3], vals[m3]
    m4 = i_mod == 4
    r[m4], g[m4], b[m4] = t[m4], p[m4], vals[m4]
    m5 = i_mod == 5
    r[m5], g[m5], b[m5] = vals[m5], p[m5], q[m5]

    rgb = np.stack([r, g, b], axis=1)
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    min_max_channel = 25
    too_dark = rgb_u8.max(axis=1) < min_max_channel
    if np.any(too_dark):
        rgb_u8[too_dark] = np.random.default_rng(seed + 1).integers(
            low=0, high=255, size=(int(too_dark.sum()), 3), dtype=np.uint8
        )

    palette[0] = [0, 0, 0]
    palette[1:] = rgb_u8
    return palette


class _DSU:
    def __init__(self, nodes: List[int]):
        self.parent = {int(n): int(n) for n in nodes}

    def find(self, x: int) -> int:
        px = self.parent.get(x, x)
        if px != x:
            self.parent[x] = self.find(px)
        return self.parent.get(x, x)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def _collect_candidate_ids(edges: List[Dict]) -> List[int]:
    ids = set()
    for e in edges:
        ids.add(int(e["u"]))
        ids.add(int(e["v"]))
    return sorted(ids)


def _build_merge_mapping(candidate_ids: List[int]) -> Dict[int, int]:
    dsu = _DSU(candidate_ids)
    candidate_set = set(candidate_ids)

    for a, b in MERGE_PAIRS:
        if a in candidate_set and b in candidate_set:
            dsu.union(int(a), int(b))
        else:
            print(f"[Merge] Skip pair ({a}, {b}) not in candidate_graph instances.")

    groups: Dict[int, List[int]] = {}
    for idx in candidate_ids:
        root = dsu.find(int(idx))
        groups.setdefault(root, []).append(int(idx))

    mapping: Dict[int, int] = {}
    for members in groups.values():
        merged_id = int(min(members))
        for m in members:
            mapping[int(m)] = merged_id
    return mapping


def _export_merged_masks(
    view_data: List[Dict],
    mapping: Dict[int, int],
    output_dir: Path,
    vis_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    max_id = max(mapping.values()) if mapping else 0
    if max_id > 254:
        print(
            f"[Warn] merged_id max={max_id} exceeds uint8 range. "
            "Values will be clipped to 254 in mask outputs."
        )
    palette = _generate_contrast_palette(min(max_id, 254) + 2, seed=42)

    for view in tqdm(view_data, desc="[Export] merged_sam"):
        mask = read_mask(view["mask_path"])
        if mask is None:
            continue

        merged = np.full(mask.shape, 255, dtype=np.uint8)
        valid = mask != 255
        if np.any(valid):
            max_label = int(mask[valid].max())
            lut = np.full(max_label + 1, 255, dtype=np.uint8)
            for src_id, dst_id in mapping.items():
                if src_id <= max_label:
                    lut[int(src_id)] = np.uint8(min(int(dst_id), 254))
            merged[valid] = lut[mask[valid]]

        img_name = Path(view["image_name"]).stem
        cv2.imwrite(str(output_dir / f"{img_name}.png"), merged)

        vis = np.zeros((merged.shape[0], merged.shape[1], 3), dtype=np.uint8)
        present = merged != 255
        if np.any(present):
            indices = merged[present].astype(np.int64) + 1
            indices = np.clip(indices, 1, len(palette) - 1)
            vis[present] = palette[indices]
        cv2.imwrite(str(vis_dir / f"{img_name}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])


def select_multi_views_for_instance(
    instance_id: int,
    view_data: List[Dict],
    merged_mask_dir: Path,
    *,
    top_k: int = 5,
    min_frame_dist: int = 20,
) -> List[Dict]:
    candidates: List[Dict] = []

    for frame_pos, view in enumerate(view_data):
        frame_id = frame_pos + 1
        mask_path = merged_mask_dir / f"{view['image_name']}.png"
        full_mask = read_mask(mask_path)
        if full_mask is None:
            continue

        mask_bool = full_mask == int(instance_id)
        area = int(np.count_nonzero(mask_bool))
        if area < 50:
            continue

        h, w = full_mask.shape
        img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        center, touching = get_mask_properties(mask_bool)
        if center is None:
            continue

        dx = abs(float(center[0]) - float(img_center[0])) / max(float(w) / 2.0, 1.0)
        dy = abs(float(center[1]) - float(img_center[1])) / max(float(h) / 2.0, 1.0)
        edge_norm = min(max(dx, dy), 1.0)
        centering_factor = 0.2 + 0.8 * ((1.0 - edge_norm) ** 2)
        boundary_factor = 0.1 if touching else 1.0
        final_score = float(area) * float(centering_factor) * float(boundary_factor)

        candidates.append(
            {
                "frame_idx": frame_id,
                "frame_pos": frame_pos,
                "score": float(final_score),
                "area": area,
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)

    selected_views: List[Dict] = []
    selected_indices: List[int] = []
    for cand in candidates:
        if len(selected_views) >= top_k:
            break
        current_idx = int(cand["frame_idx"])
        if any(abs(current_idx - picked_idx) < min_frame_dist for picked_idx in selected_indices):
            continue
        selected_views.append(cand)
        selected_indices.append(current_idx)

    return selected_views


def _export_best_views(
    instance_ids: List[int],
    view_data: List[Dict],
    merged_mask_dir: Path,
    output_dir: Path,
    *,
    top_k: int = 5,
    min_frame_dist: int = 20,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for instance_id in tqdm(instance_ids, desc="[Export] multi views"):
        views = select_multi_views_for_instance(
            instance_id,
            view_data,
            merged_mask_dir,
            top_k=top_k,
            min_frame_dist=min_frame_dist,
        )
        if not views:
            continue

        for rank, view_info in enumerate(views):
            frame_idx = int(view_info["frame_idx"])
            frame_pos = int(view_info["frame_pos"])
            view = view_data[frame_pos]
            img = cv2.imread(str(view["image_path"]))
            if img is None:
                continue

            mask_path = merged_mask_dir / f"{view['image_name']}.png"
            merged_mask = read_mask(mask_path)
            if merged_mask is None:
                continue
            mask_bool = merged_mask == int(instance_id)
            alpha = np.zeros(merged_mask.shape, dtype=np.uint8)
            alpha[mask_bool] = 255
            bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = alpha
            out_path = (
                output_dir / f"instance_{instance_id}_rank_{rank}_frame_{frame_idx}.png"
            )
            cv2.imwrite(str(out_path), bgra)


def _export_merged_ply(
    gaussian_means: torch.Tensor,
    total_point_ids_list: List[np.ndarray],
    mapping: Dict[int, int],
    output_path: Path,
) -> None:
    if not mapping:
        print("[PLY] No candidate instances to export.")
        return

    xyz = gaussian_means.detach().cpu().numpy()
    merged_to_points: Dict[int, List[int]] = {}
    for src_id, dst_id in mapping.items():
        if src_id >= len(total_point_ids_list):
            continue
        pts = np.asarray(total_point_ids_list[src_id], dtype=np.int64)
        if pts.size == 0:
            continue
        merged_to_points.setdefault(int(dst_id), []).append(pts)

    if not merged_to_points:
        print("[PLY] No points found for candidate instances.")
        return

    max_id = max(merged_to_points.keys())
    palette = _generate_contrast_palette(min(max_id, 254) + 2, seed=42)

    all_pts: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []
    for merged_id, pts_list in merged_to_points.items():
        merged_pts = np.unique(np.concatenate(pts_list, axis=0))
        if merged_pts.size == 0:
            continue
        color_u8 = palette[min(int(merged_id), 254) + 1]
        pts = xyz[merged_pts]
        color = (color_u8.astype(np.float32) / 255.0)[None, :]
        all_pts.append(pts)
        all_cols.append(np.repeat(color, pts.shape[0], axis=0))

    if not all_pts:
        print("[PLY] No merged points to write.")
        return

    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcld.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(str(output_path), pcld)
    print(f"[PLY] Saved merged instance point cloud to {output_path}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    cluster_dir = data_dir / "cluster_result"

    graph_path = cluster_dir / "candidate_graph.pt"
    tracking_path = cluster_dir / "gauscluster_tracking_data.pt"
    if not tracking_path.exists():
        tracking_path = data_dir / "gauscluster_tracking_data.pt"

    if not graph_path.exists():
        raise FileNotFoundError(f"candidate_graph.pt not found: {graph_path}")
    if not tracking_path.exists():
        raise FileNotFoundError(f"tracking data not found: {tracking_path}")

    print(f"[Load] {graph_path}")
    graph = torch.load(graph_path)
    print(f"[Load] {tracking_path}")
    tracking = torch.load(tracking_path)

    edges = graph.get("edges", [])
    candidate_ids = _collect_candidate_ids(edges)
    print(f"[Merge] candidate instances: {len(candidate_ids)}")

    mapping = _build_merge_mapping(candidate_ids)
    print(f"[Merge] merge groups: {len(set(mapping.values()))}")

    view_data = tracking["view_data"]
    merged_dir = cluster_dir / "merged_sam"
    merged_vis_dir = cluster_dir / "merged_sam_vis"
    _export_merged_masks(view_data, mapping, merged_dir, merged_vis_dir)
    _export_best_views(
        sorted(set(mapping.values())),
        view_data,
        merged_dir,
        cluster_dir / "instance_best_views",
        top_k=3,
        min_frame_dist=3,
    )

    if "gaussian_means" in tracking and "total_point_ids_list" in tracking:
        _export_merged_ply(
            tracking["gaussian_means"],
            tracking["total_point_ids_list"],
            mapping,
            cluster_dir / "merged_instances.ply",
        )
    else:
        print("[PLY] tracking data missing gaussian_means/total_point_ids_list; skip PLY.")

    print("[Done] Merged outputs saved to:")
    print(f"  - {merged_dir}")
    print(f"  - {merged_vis_dir}")
    print(f"  - {cluster_dir / 'instance_best_views'}")
    print(f"  - {cluster_dir / 'merged_instances.ply'}")


if __name__ == "__main__":
    main()
