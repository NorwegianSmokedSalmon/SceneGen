import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def read_mask(mask_path: str | Path) -> np.ndarray | None:
    """Read a mask file (supports .npy or image formats)."""
    mask_path = str(mask_path)
    if mask_path.endswith(".npy"):
        try:
            mask = np.load(mask_path)
        except Exception as exc:
            print(f"Error loading npy mask {mask_path}: {exc}")
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
    """
    Compute mask center (bbox center) and whether the bbox touches image border.

    Returns:
        (center_xy, is_touching_border)
        - center_xy is a float array [x, y] in pixel coordinates, or None if mask empty.
        - is_touching_border is True if bbox is within `border_margin_px` of any edge.
    """
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


def select_best_view_by_mask(uid_a, uid_b, instance_data, view_data):
    """
    Strategy: Considers Area + Centering + Completeness.

    Final Score = Base Score * Centering Factor * Boundary Factor
    - Base Score: min(area_a, area_b)
    - Centering Factor: penalizes views where the *smaller* instance center is near any image border (0.1~1.0)
    - Boundary Factor: penalizes views where the *smaller* instance bbox touches image border.

    Assumptions (new pipeline only):
    - `view_data[*]["mask_path"]` points to `filtered_sam/*.png` where each pixel stores `instance_id`
      and background is 255.
    - `instance_data[instance_id]` is a list of (frame_idx, instance_id) tuples.
    """
    masks_a_list = instance_data.get(uid_a, [])
    masks_b_list = instance_data.get(uid_b, [])

    def _parse_frames(uid, entries):
        frames = set()
        for entry in entries:
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                raise ValueError(
                    "Expected instance_to_masks entries as (frame_idx, instance_id) tuples "
                    f"for instance {uid}, got: {entry!r}"
                )
            frame_idx = int(entry[0])
            inst_id = int(entry[1])
            if inst_id != int(uid):
                raise ValueError(
                    "Expected instance_to_masks entries as (frame_idx, instance_id) with instance_id==uid; "
                    f"got uid={uid}, entry={entry!r}"
                )
            frames.add(frame_idx)
        return frames

    frames_a = _parse_frames(uid_a, masks_a_list)
    frames_b = _parse_frames(uid_b, masks_b_list)
    common_frames = list(frames_a.intersection(frames_b))
    if not common_frames:
        return None

    best = None
    best_score = -1.0
    for frame_idx in common_frames:
        view_info = view_data[frame_idx]
        full_mask = read_mask(view_info["mask_path"])
        if full_mask is None:
            continue

        h, w = full_mask.shape
        img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        mask_id_a = int(uid_a)
        mask_id_b = int(uid_b)

        mask_a_bool = full_mask == mask_id_a
        mask_b_bool = full_mask == mask_id_b

        area_a = int(np.count_nonzero(mask_a_bool))
        area_b = int(np.count_nonzero(mask_b_bool))

        base_score = min(area_a, area_b)
        if base_score < 50:
            continue

        center_a, touch_a = get_mask_properties(mask_a_bool)
        center_b, touch_b = get_mask_properties(mask_b_bool)
        if center_a is None or center_b is None:
            continue

        # Only use the smaller instance for centering.
        if area_a < area_b:
            centering_point = center_a
        elif area_b < area_a:
            centering_point = center_b
        else:
            centering_point = (center_a + center_b) / 2.0

        dx = abs(float(centering_point[0]) - float(img_center[0])) / max(float(w) / 2.0, 1.0)
        dy = abs(float(centering_point[1]) - float(img_center[1])) / max(float(h) / 2.0, 1.0)
        edge_norm = min(max(dx, dy), 1.0)  # 0 at center, 1 at border
        centering_factor = 0.1 + 0.9 * ((1.0 - edge_norm) ** 2)

        # Only penalize border-touching if the smaller instance is touching the border.
        if area_a < area_b:
            is_touching = touch_a
        elif area_b < area_a:
            is_touching = touch_b
        else:
            is_touching = touch_a or touch_b

        boundary_factor = 0.1 if is_touching else 1.0

        final_score = float(base_score) * float(centering_factor) * float(boundary_factor)

        if final_score > best_score:
            best_score = float(final_score)
            best = {
                "frame_idx": frame_idx,
                "score": float(final_score),
                "mask_id_a": mask_id_a,
                "mask_id_b": mask_id_b,
                "full_mask": full_mask,
            }

    return best


def get_optimal_label_pos(mask_bool: np.ndarray) -> tuple[int, int] | None:
    """
    Find the point inside the mask farthest from any boundary (distance transform).
    Returns (x, y) or None if mask is empty.
    """
    if not np.any(mask_bool):
        return None
    mask_u8 = mask_bool.astype(np.uint8)
    dist_map = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_map)
    if max_val <= 0:
        return None
    return int(max_loc[0]), int(max_loc[1])


def _place_text_origin(
    center_xy: tuple[int, int],
    text: str,
    font: int,
    font_scale: float,
    thickness: int,
    image_shape: tuple[int, int, int],
) -> tuple[int, int]:
    """Place text centered at center_xy while keeping the text inside the image."""
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = int(image_shape[0]), int(image_shape[1])

    x = int(center_xy[0] - text_w / 2)
    y = int(center_xy[1] + text_h / 2)

    x = max(0, min(x, w - text_w - 1))
    y = max(text_h + 1, min(y, h - baseline - 1))
    return x, y


def _get_bbox(mask_bool: np.ndarray) -> tuple[int, int, int, int] | None:
    if not np.any(mask_bool):
        return None
    rows, cols = np.where(mask_bool)
    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(cols.min()), int(cols.max())
    return x_min, y_min, x_max, y_max


def _place_text_above_bbox(
    bbox: tuple[int, int, int, int],
    text: str,
    font: int,
    font_scale: float,
    thickness: int,
    image_shape: tuple[int, int, int],
) -> tuple[int, int]:
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = int(image_shape[0]), int(image_shape[1])
    x_min, y_min, x_max, _ = bbox
    x = int((x_min + x_max) / 2 - text_w / 2)
    x = max(0, min(x, w - text_w - 1))
    y = y_min - 3
    y = max(text_h + 1, y)
    y = min(y, h - baseline - 1)
    return x, y


def visualize_instance_masks(
    image_bgr: np.ndarray,
    full_mask: np.ndarray,
    mask_id_a: int,
    mask_id_b: int,
    alpha: float = 0.2,
) -> np.ndarray:
    """
    Draw bounding boxes and IDs.
    Smaller instance: Red, larger instance: Green.
    """
    if image_bgr is None or full_mask is None:
        return image_bgr

    vis_img = image_bgr.copy()
    color_red = (0, 0, 255)  # BGR red
    color_green = (0, 255, 0)  # BGR green

    mask_a_bool = full_mask == int(mask_id_a)
    mask_b_bool = full_mask == int(mask_id_b)

    area_a = int(np.count_nonzero(mask_a_bool))
    area_b = int(np.count_nonzero(mask_b_bool))

    if area_a <= area_b:
        red_mask = mask_a_bool
        green_mask = mask_b_bool
        red_id = int(mask_id_a)
        green_id = int(mask_id_b)
    else:
        red_mask = mask_b_bool
        green_mask = mask_a_bool
        red_id = int(mask_id_b)
        green_id = int(mask_id_a)

    def _draw_bbox(mask_bool: np.ndarray, color: tuple[int, int, int]) -> None:
        bbox = _get_bbox(mask_bool)
        if bbox is None:
            return
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), color, 2)

    _draw_bbox(green_mask, color_green)
    _draw_bbox(red_mask, color_red)

    # SoM-style label placement: process small->large and subtract previous masks.
    label_specs = [
        {
            "mask": red_mask,
            "color": color_red,
            "id": red_id,
            "area": area_a if red_id == mask_id_a else area_b,
            "bbox": _get_bbox(red_mask),
        },
        {
            "mask": green_mask,
            "color": color_green,
            "id": green_id,
            "area": area_b if green_id == mask_id_b else area_a,
            "bbox": _get_bbox(green_mask),
        },
    ]
    label_specs.sort(key=lambda x: x["area"])

    used_mask = np.zeros_like(full_mask, dtype=bool)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2

    for spec in label_specs:
        mask_bool = spec["mask"]
        if not np.any(mask_bool):
            continue
        effective_mask = mask_bool & ~used_mask
        if not np.any(effective_mask):
            effective_mask = mask_bool
        pos = get_optimal_label_pos(effective_mask)
        used_mask |= mask_bool
        if pos is None:
            continue
        text = str(spec["id"])
        bbox = spec["bbox"]
        if bbox is not None:
            (text_w, text_h), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            bbox_w = max(1, bbox[2] - bbox[0] + 1)
            bbox_h = max(1, bbox[3] - bbox[1] + 1)
            too_big = (text_w > 0.8 * bbox_w) or (text_h > 0.5 * bbox_h)
        else:
            too_big = False
        if bbox is not None and too_big:
            text_pos = _place_text_above_bbox(
                bbox, text, font, font_scale, thickness, vis_img.shape
            )
        else:
            text_pos = _place_text_origin(
                pos, text, font, font_scale, thickness, vis_img.shape
            )
        cv2.putText(
            vis_img,
            text,
            text_pos,
            font,
            font_scale,
            (0, 0, 0),
            thickness + 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis_img,
            text,
            text_pos,
            font,
            font_scale,
            spec["color"],
            thickness,
            cv2.LINE_AA,
        )

    return vis_img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=Path, required=True, help="Dataset root directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    cluster_dir = data_dir / "cluster_result"

    print("[Load] Loading graph...")
    graph = torch.load(cluster_dir / "candidate_graph.pt")

    print("[Load] Loading tracking data...")
    tracking_path = cluster_dir / "gauscluster_tracking_data.pt"
    if not tracking_path.exists():
        tracking_path = data_dir / "gauscluster_tracking_data.pt"
    tracking = torch.load(tracking_path)

    view_data = tracking["view_data"]
    instance_to_masks = tracking["instance_to_masks"]

    output_dir = cluster_dir / "vlm_views"
    output_dir.mkdir(exist_ok=True)

    edges = graph["edges"]
    print(f"[Process] Selecting views for {len(edges)} pairs using MASKS...")

    for edge in tqdm(edges):
        u, v = edge["u"], edge["v"]

        best_view = select_best_view_by_mask(u, v, instance_to_masks, view_data)
        if not best_view:
            continue

        frame_idx = best_view["frame_idx"]
        mask_id_a = best_view["mask_id_a"]
        mask_id_b = best_view["mask_id_b"]
        view_info = view_data[frame_idx]

        img = cv2.imread(str(view_info["image_path"]))
        full_mask = best_view.get("full_mask")
        if full_mask is None:
            full_mask = read_mask(view_info["mask_path"])
        if img is None or full_mask is None:
            continue

        vis_img = visualize_instance_masks(
            img, full_mask, mask_id_a, mask_id_b, alpha=0.2
        )
        save_name = f"{u}_{v}_f{frame_idx + 1}.jpg"
        cv2.imwrite(str(output_dir / save_name), vis_img)

    print(f"[Done] Images saved to {output_dir}")


if __name__ == "__main__":
    main()
