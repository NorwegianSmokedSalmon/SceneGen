import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


_IMG_RE = re.compile(r"^instance_(\d+)_rank_(\d+)\.png$", re.IGNORECASE)
_MASK_RE = re.compile(r"^instance_(\d+)_rank_(\d+)(?:_mask)?\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class ViewEntry:
    instance_id: int
    rank: int
    image_path: Path
    mask_path: Optional[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble per-instance multi-view templates from "
            "cluster_result/candidate_views/{images,samrefiner}."
        )
    )
    parser.add_argument(
        "--candidate_views_dir",
        type=str,
        required=True,
        help="Path to cluster_result/candidate_views (contains images/ and samrefiner/).",
    )
    parser.add_argument("--images_subdir", type=str, default="images")
    parser.add_argument("--mask_subdir", type=str, default="samrefiner")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_templates",
        help="Where to write assembled template images.",
    )
    parser.add_argument(
        "--max_views",
        type=int,
        default=3,
        help="Max views per instance (sorted by rank).",
    )
    parser.add_argument(
        "--instance_id",
        type=int,
        default=None,
        help="Only process one instance id (default: all).",
    )
    parser.add_argument(
        "--cell_size",
        type=int,
        default=1024,
        help="Resize each view image/mask to this square size.",
    )
    parser.add_argument(
        "--col_padding",
        type=int,
        default=32,
        help="Horizontal padding between view columns.",
    )
    parser.add_argument(
        "--col_divider_width",
        type=int,
        default=6,
        help="Black divider line width between columns (pixels).",
    )
    parser.add_argument(
        "--arrow_height",
        type=int,
        default=96,
        help="Height of the arrow separator between top/bottom rows.",
    )
    parser.add_argument(
        "--bottom_min_bbox_area_ratio",
        type=float,
        default=0.25,
        help=(
            "If the mask bbox is too small, crop+resize the bottom-row view so that "
            "bbox_area / image_area is at least this ratio (default: 0.25)."
        ),
    )
    parser.add_argument(
        "--draw_bbox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw a red bbox on the top-row images (requires masks).",
    )
    parser.add_argument(
        "--bbox_padding",
        type=int,
        default=10,
        help="Extra pixels around mask bbox when drawing the red box.",
    )
    parser.add_argument(
        "--bbox_thickness",
        type=int,
        default=8,
        help="Red bbox line thickness.",
    )
    parser.add_argument(
        "--min_mask_pixels",
        type=int,
        default=50,
        help="Treat mask as empty if it has fewer foreground pixels than this.",
    )
    parser.add_argument(
        "--require_mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, skip any view without a corresponding mask.",
    )
    parser.add_argument(
        "--draw_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw small text labels on each column (instance id + rank).",
    )
    parser.add_argument(
        "--output_ext",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image format.",
    )
    parser.add_argument(
        "--write_manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a manifest JSON with selected views and paths.",
    )
    return parser.parse_args()


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {img.shape} for {path}")
    return img


def _read_bgr_with_src_size(path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    src_h, src_w = int(img.shape[0]), int(img.shape[1])
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {img.shape} for {path}")
    return img, (src_w, src_h)


def _read_alpha(mask_path: Path) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    if m.ndim == 3 and m.shape[2] == 4:
        return m[..., 3]
    if m.ndim == 3:
        return m[..., 0]
    return m


def _read_alpha_with_src_size(mask_path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    src_h, src_w = int(m.shape[0]), int(m.shape[1])
    if m.ndim == 3 and m.shape[2] == 4:
        return m[..., 3], (src_w, src_h)
    if m.ndim == 3:
        return m[..., 0], (src_w, src_h)
    return m, (src_w, src_h)


def _resize_square(img: np.ndarray, size: int, *, is_mask: bool) -> np.ndarray:
    if int(size) <= 0:
        return img
    h, w = int(img.shape[0]), int(img.shape[1])
    if h == size and w == size:
        return img
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(img, (int(size), int(size)), interpolation=interp)


def _bbox_from_alpha(alpha_u8: np.ndarray, *, padding: int, min_pixels: int) -> Optional[Tuple[int, int, int, int]]:
    mask_bool = alpha_u8.astype(np.uint8, copy=False) > 0
    if int(np.count_nonzero(mask_bool)) < int(min_pixels):
        return None
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return None
    h, w = int(alpha_u8.shape[0]), int(alpha_u8.shape[1])
    x1 = max(int(xs.min()) - int(padding), 0)
    y1 = max(int(ys.min()) - int(padding), 0)
    x2 = min(int(xs.max()) + int(padding), w - 1)
    y2 = min(int(ys.max()) + int(padding), h - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _draw_label(img: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 10
    x1, y1 = pad, pad
    x2, y2 = pad + tw + 2 * pad, pad + th + 2 * pad
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
    cv2.putText(img, text, (x1 + pad, y2 - pad), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _composite_on_white(img_bgr: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    img_f = img_bgr.astype(np.float32)
    alpha = (alpha_u8.astype(np.float32) / 255.0).reshape(img_bgr.shape[0], img_bgr.shape[1], 1)
    white = np.full_like(img_f, 255.0)
    out = img_f * alpha + white * (1.0 - alpha)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _square_crop_params_for_bbox(
    *,
    bbox_xyxy: Tuple[int, int, int, int],
    image_size: int,
    min_bbox_area_ratio: float,
) -> Optional[Tuple[int, int, int]]:
    """
    Compute a square crop (x0, y0, side) that:
      - fully contains bbox
      - makes bbox occupy at least `min_bbox_area_ratio` of the crop area (approximately),
        which after resize-to-square preserves bbox_area/crop_area.
    Returns None if no crop needed.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    w = int(x2 - x1 + 1)
    h = int(y2 - y1 + 1)
    if w <= 0 or h <= 0:
        return None

    bbox_area = float(w * h)
    img_area = float(image_size * image_size)
    if img_area <= 0:
        return None

    ratio_now = bbox_area / img_area
    target = float(min_bbox_area_ratio)
    if not np.isfinite(target) or target <= 0.0:
        return None
    if ratio_now >= target:
        return None

    # Want: bbox_area / crop_area >= target  =>  crop_side^2 <= bbox_area / target
    side_limit = math.sqrt(max(1.0, bbox_area / max(target, 1e-12)))
    side = int(math.floor(side_limit))
    side = max(side, int(max(w, h)))
    side = min(side, int(image_size))
    if side >= int(image_size):
        return None

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    # Ensure crop fully contains bbox: x0 in [x2-side+1, x1], clamped to image bounds.
    x0_min = max(0, int(x2 - side + 1))
    x0_max = min(int(image_size - side), int(x1))
    if x0_min > x0_max:
        return None
    x0 = int(round(cx - side * 0.5))
    x0 = max(x0_min, min(x0, x0_max))

    y0_min = max(0, int(y2 - side + 1))
    y0_max = min(int(image_size - side), int(y1))
    if y0_min > y0_max:
        return None
    y0 = int(round(cy - side * 0.5))
    y0 = max(y0_min, min(y0, y0_max))

    return x0, y0, side


def _build_column_separator(
    *,
    height: int,
    col_padding: int,
    col_divider_width: int,
) -> np.ndarray:
    divider_w = max(1, int(col_divider_width))
    gap_w = int(col_padding)
    if gap_w <= 0:
        gap_left = 0
        gap_right = 0
    else:
        sep_w = max(gap_w, divider_w)
        remaining = sep_w - divider_w
        gap_left = remaining // 2
        gap_right = remaining - gap_left

    parts: List[np.ndarray] = []
    if gap_left > 0:
        parts.append(np.full((int(height), int(gap_left), 3), 255, dtype=np.uint8))
    parts.append(np.full((int(height), int(divider_w), 3), 0, dtype=np.uint8))
    if gap_right > 0:
        parts.append(np.full((int(height), int(gap_right), 3), 255, dtype=np.uint8))
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def _pad_to_aspect_ratio(
    img: np.ndarray,
    *,
    aspect_w: int,
    aspect_h: int,
    pad_value: int = 255,
) -> Tuple[np.ndarray, int, int]:
    """
    Pad (no crop) an image to a target aspect ratio (aspect_w:aspect_h).
    Returns (padded_img, pad_left, pad_top).
    """
    h, w = int(img.shape[0]), int(img.shape[1])
    if h <= 0 or w <= 0:
        return img, 0, 0
    aspect = float(aspect_w) / float(aspect_h)
    cur = float(w) / float(h)

    if abs(cur - aspect) < 1e-6:
        return img, 0, 0

    if cur < aspect:
        # Need wider canvas.
        new_w = int(math.ceil(float(h) * aspect))
        new_h = h
    else:
        # Need taller canvas.
        new_w = w
        new_h = int(math.ceil(float(w) / aspect))

    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left
    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top

    padded = np.full((new_h, new_w, img.shape[2]), int(pad_value), dtype=img.dtype)
    padded[pad_top : pad_top + h, pad_left : pad_left + w] = img
    return padded, int(pad_left), int(pad_top)


def _draw_down_arrow(block: np.ndarray) -> None:
    h, w = int(block.shape[0]), int(block.shape[1])
    x = w // 2
    y1 = max(0, int(h * 0.2))
    y2 = min(h - 1, int(h * 0.8))
    thickness = max(14, int(round(h * 0.18)))
    tip_len = max(22, int(round(h * 0.32)))
    tip_half_w = max(thickness * 3, int(round(tip_len * 1.25)))

    cv2.line(block, (x, y1), (x, max(y1, y2 - tip_len)), (0, 0, 0), thickness=thickness)
    tip_y = y2
    base_y = max(y1, y2 - tip_len)
    tri = np.array(
        [
            [x, tip_y],
            [x - tip_half_w, base_y],
            [x + tip_half_w, base_y],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(block, tri, (0, 0, 0))


def _mask_path_for(masks_dir: Path, instance_id: int, rank: int) -> Optional[Path]:
    direct = [
        masks_dir / f"instance_{int(instance_id)}_rank_{int(rank):02d}_mask.png",
        masks_dir / f"instance_{int(instance_id)}_rank_{int(rank)}_mask.png",
        # SAM3 refinement stores RGBA masks under the original image name.
        masks_dir / f"instance_{int(instance_id)}_rank_{int(rank):02d}.png",
        masks_dir / f"instance_{int(instance_id)}_rank_{int(rank)}.png",
    ]
    for p in direct:
        if p.exists():
            return p
    for p in masks_dir.glob(f"instance_{int(instance_id)}_rank_*.png"):
        m = _MASK_RE.match(p.name)
        if not m:
            continue
        if int(m.group(1)) == int(instance_id) and int(m.group(2)) == int(rank):
            return p
    return None


def _collect_views(
    images_dir: Path, masks_dir: Path, *, require_mask: bool, instance_id: Optional[int]
) -> Dict[int, List[ViewEntry]]:
    grouped: Dict[int, List[ViewEntry]] = {}
    for img_path in images_dir.glob("instance_*_rank_*.png"):
        m = _IMG_RE.match(img_path.name)
        if not m:
            continue
        inst = int(m.group(1))
        if instance_id is not None and int(inst) != int(instance_id):
            continue
        rank = int(m.group(2))
        mask_path = _mask_path_for(masks_dir=masks_dir, instance_id=inst, rank=rank)
        if require_mask and mask_path is None:
            continue
        grouped.setdefault(inst, []).append(
            ViewEntry(instance_id=inst, rank=rank, image_path=img_path, mask_path=mask_path)
        )
    for inst, views in grouped.items():
        grouped[inst] = sorted(views, key=lambda v: int(v.rank))
    return grouped


def _assemble_instance_template(
    *,
    instance_id: int,
    views: List[ViewEntry],
    target_num_columns: int,
    cell_size: int,
    col_padding: int,
    col_divider_width: int,
    arrow_height: int,
    draw_bbox: bool,
    bottom_min_bbox_area_ratio: float,
    bbox_padding: int,
    bbox_thickness: int,
    min_mask_pixels: int,
    draw_labels: bool,
) -> Tuple[np.ndarray, List[Dict]]:
    cols: List[np.ndarray] = []
    manifest_views: List[Dict] = []

    col_height = int(cell_size) * 2 + int(arrow_height)
    col_sep = _build_column_separator(
        height=col_height,
        col_padding=int(col_padding),
        col_divider_width=int(col_divider_width),
    )
    sep_w = int(col_sep.shape[1])
    bottom_y = int(cell_size) + int(arrow_height)

    for col_idx, v in enumerate(views):
        img, (src_w, src_h) = _read_bgr_with_src_size(v.image_path)
        img = _resize_square(img, cell_size, is_mask=False)

        alpha_u8: Optional[np.ndarray] = None
        bbox: Optional[Tuple[int, int, int, int]] = None
        mask_src_w: Optional[int] = None
        mask_src_h: Optional[int] = None
        if v.mask_path is not None and v.mask_path.exists():
            alpha_u8, (mask_src_w, mask_src_h) = _read_alpha_with_src_size(v.mask_path)
            alpha_u8 = _resize_square(alpha_u8, cell_size, is_mask=True)
            bbox = _bbox_from_alpha(alpha_u8, padding=bbox_padding, min_pixels=min_mask_pixels)

        top = img.copy()
        if bool(draw_bbox) and bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(top, (x1, y1), (x2, y2), (0, 0, 255), thickness=int(bbox_thickness))

        if draw_labels:
            _draw_label(top, f"id={int(instance_id)} rank={int(v.rank):02d}")

        crop_x0 = 0
        crop_y0 = 0
        crop_side = int(cell_size)
        crop_applied = False

        if alpha_u8 is None:
            bottom = img.copy()
        else:
            crop = None
            if bbox is not None:
                crop = _square_crop_params_for_bbox(
                    bbox_xyxy=bbox,
                    image_size=int(cell_size),
                    min_bbox_area_ratio=float(bottom_min_bbox_area_ratio),
                )
            if crop is not None:
                crop_x0, crop_y0, crop_side = [int(x) for x in crop]
                crop_applied = True
                img_crop = img[crop_y0 : crop_y0 + crop_side, crop_x0 : crop_x0 + crop_side]
                alpha_crop = alpha_u8[crop_y0 : crop_y0 + crop_side, crop_x0 : crop_x0 + crop_side]
                img_crop = _resize_square(img_crop, cell_size, is_mask=False)
                alpha_crop = _resize_square(alpha_crop, cell_size, is_mask=True)
                bottom = _composite_on_white(img_bgr=img_crop, alpha_u8=alpha_crop)
            else:
                bottom = _composite_on_white(img_bgr=img, alpha_u8=alpha_u8)

        sep = np.full((int(arrow_height), int(cell_size), 3), 255, dtype=np.uint8)
        _draw_down_arrow(sep)

        col = np.vstack([top, sep, bottom])
        cols.append(col)

        x_offset = int(col_idx) * (int(cell_size) + int(sep_w))
        top_xywh = {"x": x_offset, "y": 0, "w": int(cell_size), "h": int(cell_size)}
        bottom_xywh = {"x": x_offset, "y": int(bottom_y), "w": int(cell_size), "h": int(cell_size)}

        sx = float(cell_size) / float(max(1, int(src_w)))
        sy = float(cell_size) / float(max(1, int(src_h)))

        manifest_views.append(
            {
                "col_index": int(col_idx),
                "rank": int(v.rank),
                "image": str(v.image_path.name),
                "mask": str(v.mask_path.name) if v.mask_path is not None else None,
                "bbox_xyxy": [int(x) for x in bbox] if bbox is not None else None,
                "template_top_xywh": top_xywh,
                "template_bottom_xywh": bottom_xywh,
                "bottom_crop_base": {
                    "x0": int(crop_x0),
                    "y0": int(crop_y0),
                    "side": int(crop_side),
                    "applied": bool(crop_applied),
                },
                "src_wh": {"w": int(src_w), "h": int(src_h)},
                "mask_src_wh": (
                    {"w": int(mask_src_w), "h": int(mask_src_h)}
                    if mask_src_w is not None and mask_src_h is not None
                    else None
                ),
                "base_wh": {"w": int(cell_size), "h": int(cell_size)},
                "src_to_base_scale": {"sx": float(sx), "sy": float(sy)},
                "layout": {
                    "cell_size": int(cell_size),
                    "arrow_height": int(arrow_height),
                    "col_separator_width": int(sep_w),
                    "col_padding": int(col_padding),
                    "col_divider_width": int(col_divider_width),
                },
            }
        )

    if not cols:
        raise ValueError(f"Instance {instance_id}: no views to assemble.")

    # If we have fewer columns than requested, append blank columns (keeps layout stable).
    target_cols = max(1, int(target_num_columns))
    if target_cols > len(cols):
        blank_top = np.full((int(cell_size), int(cell_size), 3), 255, dtype=np.uint8)
        blank_sep = np.full((int(arrow_height), int(cell_size), 3), 255, dtype=np.uint8)
        _draw_down_arrow(blank_sep)
        blank_bottom = np.full((int(cell_size), int(cell_size), 3), 255, dtype=np.uint8)
        blank_col = np.vstack([blank_top, blank_sep, blank_bottom])
        for _ in range(target_cols - len(cols)):
            cols.append(blank_col.copy())

    stitched: List[np.ndarray] = []
    for i, col in enumerate(cols):
        stitched.append(col)
        if i != len(cols) - 1:
            stitched.append(col_sep)
    grid = np.hstack(stitched)

    # Enforce a 3:2 output aspect ratio by padding (no crop).
    canvas, pad_left, pad_top = _pad_to_aspect_ratio(grid, aspect_w=3, aspect_h=2, pad_value=255)
    if pad_left != 0 or pad_top != 0:
        for entry in manifest_views:
            entry["template_top_xywh"]["x"] = int(entry["template_top_xywh"]["x"]) + int(pad_left)
            entry["template_top_xywh"]["y"] = int(entry["template_top_xywh"]["y"]) + int(pad_top)
            entry["template_bottom_xywh"]["x"] = int(entry["template_bottom_xywh"]["x"]) + int(pad_left)
            entry["template_bottom_xywh"]["y"] = int(entry["template_bottom_xywh"]["y"]) + int(pad_top)
            entry["layout"]["content_wh"] = {"w": int(grid.shape[1]), "h": int(grid.shape[0])}
            entry["layout"]["canvas_wh"] = {"w": int(canvas.shape[1]), "h": int(canvas.shape[0])}
            entry["layout"]["canvas_pad_xy"] = {"x": int(pad_left), "y": int(pad_top)}
            entry["layout"]["canvas_aspect_ratio"] = "3:2"
    else:
        for entry in manifest_views:
            entry["layout"]["content_wh"] = {"w": int(grid.shape[1]), "h": int(grid.shape[0])}
            entry["layout"]["canvas_wh"] = {"w": int(grid.shape[1]), "h": int(grid.shape[0])}
            entry["layout"]["canvas_pad_xy"] = {"x": 0, "y": 0}
            entry["layout"]["canvas_aspect_ratio"] = "3:2"

    return canvas, manifest_views


def main() -> None:
    args = parse_args()
    root = Path(args.candidate_views_dir)
    images_dir = root / str(args.images_subdir)
    masks_dir = root / str(args.mask_subdir)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not masks_dir.exists():
        print(f"[Warning] Missing mask dir: {masks_dir} (will assemble without masks).")

    groups = _collect_views(
        images_dir=images_dir,
        masks_dir=masks_dir,
        require_mask=bool(args.require_mask),
        instance_id=args.instance_id,
    )
    if not groups:
        if args.require_mask:
            raise FileNotFoundError(f"No image/mask pairs found in {images_dir} and {masks_dir}")
        print("[Info] No instance views found.")
        return

    manifest_out: List[Dict] = []
    inst_ids = sorted(groups.keys())
    for inst in inst_ids:
        selected = groups[inst][: max(0, int(args.max_views))]
        if not selected:
            continue
        try:
            grid, used = _assemble_instance_template(
                instance_id=int(inst),
                views=selected,
                target_num_columns=max(1, int(args.max_views)),
                cell_size=int(args.cell_size),
                col_padding=int(args.col_padding),
                col_divider_width=int(args.col_divider_width),
                arrow_height=int(args.arrow_height),
                draw_bbox=bool(args.draw_bbox),
                bottom_min_bbox_area_ratio=float(args.bottom_min_bbox_area_ratio),
                bbox_padding=int(args.bbox_padding),
                bbox_thickness=int(args.bbox_thickness),
                min_mask_pixels=int(args.min_mask_pixels),
                draw_labels=bool(args.draw_labels),
            )
        except Exception as exc:
            print(f"[Warning] Instance {inst} failed: {exc}")
            continue

        out_path = out_dir / f"template_instance_{int(inst)}.{str(args.output_ext).lower()}"
        ok = cv2.imwrite(str(out_path), grid)
        if not ok:
            print(f"[Warning] Failed to write: {out_path}")
            continue

        manifest_out.append(
            {
                "instance_id": int(inst),
                "num_views": int(len(used)),
                "output": str(out_path.name),
                "views": used,
            }
        )
        print(f"[OK] {out_path.name} (views={len(used)})")

    if args.require_mask and not manifest_out:
        raise RuntimeError("No templates were produced from the required masks")

    if bool(args.write_manifest):
        manifest_path = out_dir / "templates_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_out, f, indent=2, ensure_ascii=False)
        print(f"[OK] Wrote {manifest_path}")


if __name__ == "__main__":
    main()
