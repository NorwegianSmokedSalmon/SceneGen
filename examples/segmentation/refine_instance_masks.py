import os
import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple
import inspect
import numpy as np
import torch
import cv2
from PIL import Image
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine candidate view instance masks via SAM3 box prompt."
    )
    parser.add_argument(
        "--candidate_views_dir",
        type=str,
        required=True,
        help="Path to cluster_result/candidate_views (contains images/ and projected_mask/).",
    )
    parser.add_argument("--images_subdir", type=str, default="images")
    parser.add_argument("--mask_subdir", type=str, default="projected_mask")
    parser.add_argument("--output_subdir", type=str, default="sam3_results")
    parser.add_argument(
        "--pattern",
        type=str,
        default="instance_*_rank_01.png",
        help="Glob pattern under images_subdir.",
    )
    parser.add_argument("--instance_id", type=int, default=None, help="Only process one instance id.")
    parser.add_argument(
        "--bbox_padding",
        type=int,
        default=10,
        help="Pad bbox by this many pixels on each side (clamped to image).",
    )
    parser.add_argument(
        "--draw_bbox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw the computed bbox on the output RGBA (on RGB channels).",
    )
    parser.add_argument(
        "--draw_points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw the computed point prompts on the output RGBA (on RGB channels).",
    )
    parser.add_argument(
        "--bbox_thickness",
        type=int,
        default=3,
        help="Bounding box line thickness in pixels for visualization.",
    )
    parser.add_argument(
        "--min_mask_pixels",
        type=int,
        default=50,
        help="Skip if coarse mask has fewer foreground pixels than this.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Number of refinement iterations (subsequent iterations feed previous low-res logits as mask_input).",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable debug prints and prompt visualization (draw bbox/points).",
    )
    parser.add_argument(
        "--box_mode",
        type=str,
        default="pixel",
        choices=["auto", "normalized", "pixel"],
        help="Box format for SAM3: normalized [0,1] xyxy, or pixel xyxy. auto tries normalized then pixel.",
    )
    parser.add_argument(
        "--use_points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also prompt SAM3 with point(s) derived from the coarse mask.",
    )
    parser.add_argument(
        "--add_neg_point",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add one background point (label=0) inside the coarse-mask bbox (if available).",
    )
    parser.add_argument(
        "--point_radius",
        type=int,
        default=6,
        help="Point marker radius in pixels for visualization.",
    )
    parser.add_argument(
        "--use_mask_prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also prompt SAM3 with a low-res mask_input derived from the coarse mask (SAM1-style iterative mask prompt).",
    )
    parser.add_argument(
        "--mask_prompt_strength",
        type=float,
        default=30.0,
        help="Foreground/background logit magnitude for the derived mask_input (larger is a stronger prior).",
    )
    parser.add_argument(
        "--mask_prompt_gamma",
        type=float,
        default=4.0,
        help="Gaussian weighting factor for mask prompt (SAMRefiner-style). Larger makes weighting more concentrated.",
    )
    parser.add_argument(
        "--mask_prompt_lowres",
        type=int,
        default=None,
        help="Low-res mask_input size (e.g. 256 for 1024px models). If omitted, auto-uses model_img_size//4.",
    )
    parser.add_argument(
        "--bpe_path",
        type=str,
        default=None,
        help="Optional override for SAM3 BPE vocab path; default is derived from sam3 install.",
    )
    parser.add_argument(
        "--sam3_weights_dir",
        type=str,
        default=None,
        help="Local SAM3 weights directory (used to auto-pick a checkpoint like sam3.pt).",
    )
    parser.add_argument(
        "--sam3_ckpt_path",
        type=str,
        default=None,
        help="Optional explicit SAM3 checkpoint path (overrides --sam3_weights_dir).",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing outputs (kept for compatibility; outputs are always overwritten).",
    )
    return parser.parse_args()


def _resolve_sam3_ckpt_path(weights_dir: Optional[str], explicit_path: Optional[str]) -> Optional[str]:
    if explicit_path:
        return str(explicit_path)
    if not weights_dir:
        return None
    root = Path(weights_dir)
    if not root.exists():
        return None
    preferred = root / "sam3.pt"
    if preferred.exists():
        return str(preferred)
    for pat in ("*.pt", "*.pth", "*.ckpt", "*.safetensors"):
        matches = sorted(root.glob(pat))
        if matches:
            return str(matches[0])
    return None

def init_sam3_model(
    *,
    bpe_path: Optional[str] = None,
    sam3_weights_dir: Optional[str] = None,
    sam3_ckpt_path: Optional[str] = None,
):
    """初始化 SAM 3 模型和处理器"""
    # 选择设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"正在使用设备: {device}")

    # 针对 CUDA 的优化设置
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # 自动定位 sam3 安装目录并推导 BPE 词表路径
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    default_bpe_path = os.path.join(sam3_root, "sam3/assets", "bpe_simple_vocab_16e6.txt.gz")
    resolved_bpe_path = str(bpe_path or default_bpe_path)
    resolved_ckpt_path = _resolve_sam3_ckpt_path(
        weights_dir=sam3_weights_dir, explicit_path=sam3_ckpt_path
    )

    # 构建模型
    # 注意：如果下载了特定的 checkpoint，请查阅 sam3 文档加载权重
    build_kwargs = {
        "bpe_path": resolved_bpe_path,
        "enable_inst_interactivity": True,
    }
    if resolved_ckpt_path:
        sig = inspect.signature(build_sam3_image_model)
        param_names = set(sig.parameters.keys())
        for key in ("ckpt_path", "checkpoint_path", "weights_path", "model_path"):
            if key in param_names:
                build_kwargs[key] = resolved_ckpt_path
                break
        else:
            for key in ("ckpt_dir", "weights_dir", "model_dir"):
                if key in param_names and sam3_weights_dir:
                    build_kwargs[key] = str(sam3_weights_dir)
                    break

    model = build_sam3_image_model(**build_kwargs)
    model.to(device)

    # IMPORTANT: Sam3Processor resizes the image to a square resolution. This must match the
    # resolution expected by the interactive predictor, otherwise feature map shapes mismatch.
    processor_resolution = None
    pred = getattr(model, "inst_interactive_predictor", None)
    if pred is not None:
        inner = getattr(pred, "model", None)
        if inner is not None and hasattr(inner, "image_size"):
            try:
                processor_resolution = int(getattr(inner, "image_size"))
            except Exception:
                processor_resolution = None
    if processor_resolution is None:
        processor_resolution = 1008
    processor = Sam3Processor(model, resolution=int(processor_resolution), device=str(device))
    print(f"[SAM3] processor resolution: {processor_resolution}")
    return model, processor, device, resolved_bpe_path, resolved_ckpt_path


def _load_coarse_mask_alpha(mask_path: Path) -> np.ndarray:
    """
    Load coarse mask and return a boolean mask (H, W).
    Supports:
      - RGBA mask PNGs where alpha channel is the mask
      - single-channel PNG masks
    """
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    if m.ndim == 3 and m.shape[2] == 4:
        alpha = m[..., 3]
        return (alpha > 0)
    if m.ndim == 3:
        m = m[..., 0]
    return (m.astype(np.uint8, copy=False) > 0)


def _bbox_from_mask(mask_bool: np.ndarray, padding: int, w: int, h: int) -> np.ndarray:
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Empty mask; cannot compute bbox.")
    x1 = max(int(xs.min()) - int(padding), 0)
    y1 = max(int(ys.min()) - int(padding), 0)
    x2 = min(int(xs.max()) + int(padding), w - 1)
    y2 = min(int(ys.max()) + int(padding), h - 1)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate bbox: {(x1, y1, x2, y2)}")
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _to_numpy_bool_mask(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.dtype != np.bool_:
        mask = mask > 0.0
    return mask.astype(bool, copy=False)


def _points_from_coarse_mask(
    *,
    coarse_bool: np.ndarray,
    bbox_xyxy_px: np.ndarray,
    add_neg: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Derive SAM-style point prompts from a coarse mask:
      - one foreground point at the max of distance transform inside the mask
      - optionally one background point inside the bbox but outside the mask
    Returns:
      point_coords: (N, 2) in pixel (x, y) coordinates
      point_labels: (N,) with values {0,1}
    """
    mask_u8 = (coarse_bool.astype(np.uint8, copy=False) * 255)
    if int(mask_u8.sum()) == 0:
        return None, None

    dist_in = cv2.distanceTransform(mask_u8, distanceType=cv2.DIST_L2, maskSize=3)
    y_pos, x_pos = np.unravel_index(int(dist_in.argmax()), dist_in.shape)
    point_coords = [[int(x_pos), int(y_pos)]]
    point_labels = [1]

    if add_neg:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy_px.tolist()]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(mask_u8.shape[1] - 1, x2)
        y2 = min(mask_u8.shape[0] - 1, y2)
        if x2 > x1 and y2 > y1:
            inv_u8 = ((~coarse_bool).astype(np.uint8, copy=False) * 255)
            dist_bg = cv2.distanceTransform(inv_u8, distanceType=cv2.DIST_L2, maskSize=3)
            roi = dist_bg[y1 : y2 + 1, x1 : x2 + 1]
            roi_bg = inv_u8[y1 : y2 + 1, x1 : x2 + 1] > 0
            if bool(roi_bg.any()):
                roi_masked = np.where(roi_bg, roi, -1.0)
                y_rel, x_rel = np.unravel_index(int(roi_masked.argmax()), roi_masked.shape)
                if float(roi_masked[y_rel, x_rel]) >= 0.0:
                    point_coords.append([int(x1 + x_rel), int(y1 + y_rel)])
                    point_labels.append(0)

    return (
        np.asarray(point_coords, dtype=np.float32),
        np.asarray(point_labels, dtype=np.int64),
    )

def _get_model_img_size(model, fallback: int = 1024) -> int:
    for path in ("image_encoder.img_size", "image_encoder.image_size", "img_size", "image_size"):
        cur = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            try:
                v = int(cur)
                if v > 0:
                    return v
            except Exception:
                pass
    return int(fallback)


def _get_sam3_inst_interactive_predictor(model):
    return getattr(model, "inst_interactive_predictor", None)


def _get_sam3_prompt_resolution(model, fallback: int = 1024) -> int:
    """
    SAM3 interactive predictor uses SAM2Transforms(resolution=self.model.image_size).
    Prefer that resolution so points/boxes/masks live in the same prompt frame.
    """
    pred = _get_sam3_inst_interactive_predictor(model)
    if pred is not None:
        inner = getattr(pred, "model", None)
        if inner is not None and hasattr(inner, "image_size"):
            try:
                v = int(getattr(inner, "image_size"))
                if v > 0:
                    return v
            except Exception:
                pass
    return _get_model_img_size(model, fallback=fallback)


def _get_sam3_expected_mask_lowres(model, prompt_resolution: int) -> int:
    """
    For SAM3 inst interactive predictor, mask_input downscales to the image embedding spatial size.
    Empirically this corresponds to (mask_lowres = prompt_resolution // 4), e.g. 1152->288.
    """
    pred = _get_sam3_inst_interactive_predictor(model)
    if pred is not None and hasattr(pred, "_bb_feat_sizes"):
        try:
            bb = getattr(pred, "_bb_feat_sizes")
            if isinstance(bb, (list, tuple)) and len(bb) > 0:
                h, w = bb[0]
                h = int(h)
                w = int(w)
                if h > 0 and h == w:
                    return h
        except Exception:
            pass
    return int(prompt_resolution) // 4


def _gaussian_weight_from_mask(*, coarse_bool: np.ndarray, gamma: float) -> np.ndarray:
    """
    SAMRefiner-style weighting term for mask prompt.
    Uses distance-to-boundary (via distance transform) to down-weight logits near mask edges.

    Returns:
      weight: float32 array of shape (H, W), with 1.0 outside mask, and (0,1] inside mask.
    """
    if coarse_bool.dtype != np.bool_:
        coarse_bool = coarse_bool.astype(bool, copy=False)
    h, w = coarse_bool.shape
    if h == 0 or w == 0:
        return np.ones((h, w), dtype=np.float32)

    mask_u8 = (coarse_bool.astype(np.uint8, copy=False) * 255)
    if int(mask_u8.sum()) == 0:
        return np.ones((h, w), dtype=np.float32)

    dist_in = cv2.distanceTransform(mask_u8, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
    max_dist = float(dist_in.max())
    if max_dist <= 0.0:
        return np.ones((h, w), dtype=np.float32)

    area = float(coarse_bool.sum())
    g = float(gamma) if float(gamma) > 0.0 else 4.0
    denom = max(area / g, 1.0)
    dist0 = dist_in - max_dist  # <= 0 inside mask
    weight_inside = np.exp(-(dist0 * dist0) / denom).astype(np.float32, copy=False)

    weight = np.ones((h, w), dtype=np.float32)
    weight[coarse_bool] = weight_inside[coarse_bool]
    return weight


def _mask_to_lowres_mask_input(
    *,
    coarse_bool: np.ndarray,
    prompt_resolution: int,
    lowres_size: int,
    strength: float,
    gamma: float,
) -> np.ndarray:
    """
    Convert a coarse binary mask in original image space into a SAM-style low-res mask_input.
    The mask_input is aligned to the model input frame: resize-longest-side to model_img_size,
    pad to (model_img_size, model_img_size), then downsample to (lowres_size, lowres_size).

    Returns:
      mask_input: float32 array of shape (1, lowres_size, lowres_size)
    """
    s = float(max(1.0, strength))
    logits = np.where(coarse_bool, s, -s).astype(np.float32, copy=False)
    logits = logits * _gaussian_weight_from_mask(coarse_bool=coarse_bool, gamma=gamma)

    # SAM3 interactive predictor uses square Resize((resolution, resolution)) (aspect ratio is not preserved).
    pr = int(prompt_resolution)
    logits_pr = cv2.resize(logits, (pr, pr), interpolation=cv2.INTER_LINEAR)
    low = cv2.resize(logits_pr, (int(lowres_size), int(lowres_size)), interpolation=cv2.INTER_LINEAR)
    return low[None, :, :].astype(np.float32, copy=False)


def _predict_mask_with_prompts(
    *,
    model,
    inference_state,
    box_xyxy_px: np.ndarray,
    point_coords_px: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
    mask_input: Optional[np.ndarray],
    image_w: int,
    image_h: int,
    mode: str,
) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    box_xyxy_px = np.asarray(box_xyxy_px, dtype=np.float32).reshape(4)
    if mode == "normalized":
        x1, y1, x2, y2 = box_xyxy_px.tolist()
        input_box = np.array([[x1 / image_w, y1 / image_h, x2 / image_w, y2 / image_h]], dtype=np.float32)
    elif mode == "pixel":
        input_box = np.array([box_xyxy_px], dtype=np.float32)
    else:
        raise ValueError(f"Unknown box mode: {mode}")

    masks, scores, logits = model.predict_inst(
        inference_state,
        point_coords=point_coords_px,
        point_labels=point_labels,
        box=input_box,
        mask_input=mask_input,
        multimask_output=False,
    )
    best_mask = _to_numpy_bool_mask(masks[0])
    best_score = float(scores[0].detach().cpu().item()) if isinstance(scores, torch.Tensor) else float(scores[0])
    if logits is None:
        return best_mask, best_score, None
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits)
    if logits.ndim == 3:
        logits = logits[0]
    if logits.ndim != 2:
        return best_mask, best_score, None
    return best_mask, best_score, logits.astype(np.float32, copy=False)

def refine_from_coarse_mask(
    *,
    model,
    processor: Sam3Processor,
    image_path: Path,
    coarse_mask_path: Path,
    output_rgba_path: Path,
    bbox_padding: int,
    draw_bbox: bool,
    draw_points: bool,
    bbox_thickness: int,
    min_mask_pixels: int,
    box_mode: str,
    use_points: bool,
    add_neg_point: bool,
    point_radius: int,
    use_mask_prompt: bool,
    mask_prompt_strength: float,
    mask_prompt_gamma: float,
    mask_prompt_lowres: Optional[int],
    iters: int,
    debug: bool,
    overwrite: bool,
) -> None:
    # Always overwrite outputs (directory is cleared per run).

    image_pil = Image.open(image_path).convert("RGB")
    image_w, image_h = image_pil.size

    coarse_bool = _load_coarse_mask_alpha(coarse_mask_path)
    coarse_area = int(coarse_bool.sum())
    if coarse_area < int(min_mask_pixels):
        print(f"[Skip] {image_path.name}: coarse mask too small ({coarse_area} px)")
        return

    try:
        bbox_xyxy_px = _bbox_from_mask(
            coarse_bool, padding=bbox_padding, w=image_w, h=image_h
        )
    except Exception as exc:
        print(f"[Skip] {image_path.name}: invalid coarse mask ({exc})")
        return

    inference_state = processor.set_image(image_pil)

    point_coords_px = None
    point_labels = None
    if use_points:
        point_coords_px, point_labels = _points_from_coarse_mask(
            coarse_bool=coarse_bool,
            bbox_xyxy_px=bbox_xyxy_px,
            add_neg=add_neg_point,
        )

    mask_input = None
    if use_mask_prompt:
        prompt_resolution = _get_sam3_prompt_resolution(model, fallback=1024)
        expected_lowres = _get_sam3_expected_mask_lowres(model, prompt_resolution)
        lowres_size = int(mask_prompt_lowres) if mask_prompt_lowres is not None else int(expected_lowres)
        if debug and lowres_size != expected_lowres:
            print(
                f"[Warn] {image_path.name}: mask_input lowres={lowres_size} does not match expected {expected_lowres} "
                f"(prompt_resolution={prompt_resolution}); consider omitting --mask_prompt_lowres or setting it to {expected_lowres}."
            )
        mask_input = _mask_to_lowres_mask_input(
            coarse_bool=coarse_bool,
            prompt_resolution=prompt_resolution,
            lowres_size=lowres_size,
            strength=float(mask_prompt_strength),
            gamma=float(mask_prompt_gamma),
        )
        if debug:
            print(
                f"[Debug] {image_path.name}: prompt_resolution={prompt_resolution} mask_input={tuple(mask_input.shape)}"
            )
    if debug:
        print(f"[Debug] {image_path.name}: bbox_xyxy_px={bbox_xyxy_px.tolist()}")
        if use_points and point_coords_px is not None and point_labels is not None:
            print(
                f"[Debug] {image_path.name}: points={point_coords_px.tolist()} labels={point_labels.tolist()}"
            )

    chosen_mask = None
    chosen_score = None
    chosen_logits = None

    modes_to_try = [box_mode] if box_mode in ("normalized", "pixel") else ["normalized", "pixel"]
    effective_iters = int(max(1, iters)) if use_mask_prompt else 1
    for mode in modes_to_try:
        cur_mask_input = mask_input
        cur_mask = None
        cur_score = None
        try:
            logits_for_next = None
            for step in range(effective_iters):
                pred_mask, score, logits_out = _predict_mask_with_prompts(
                    model=model,
                    inference_state=inference_state,
                    box_xyxy_px=bbox_xyxy_px,
                    point_coords_px=point_coords_px,
                    point_labels=point_labels,
                    mask_input=cur_mask_input,
                    image_w=image_w,
                    image_h=image_h,
                    mode=mode,
                )
                cur_mask = pred_mask
                cur_score = float(score)
                logits_for_next = logits_out
                if use_mask_prompt and logits_for_next is not None:
                    cur_mask_input = logits_for_next[None, :, :].astype(np.float32, copy=False)
        except Exception as exc:
            print(f"[Warn] {image_path.name}: SAM3 predict failed in {mode} mode ({exc})")
            continue
        if cur_mask is None or cur_score is None:
            continue
        if chosen_mask is None or float(cur_score) > float(chosen_score):
            chosen_mask = cur_mask
            chosen_score = float(cur_score)
            chosen_logits = logits_for_next

    if chosen_mask is None:
        print(f"[Fail] {image_path.name}: no valid SAM3 prediction")
        return

    alpha = (chosen_mask.astype(np.uint8, copy=False) * 255).astype(np.uint8, copy=False)
    rgb = np.array(image_pil, dtype=np.uint8)
    if draw_points and use_points and point_coords_px is not None and point_labels is not None:
        r = int(max(1, point_radius))
        for (x, y), lab in zip(point_coords_px.tolist(), point_labels.tolist()):
            pt = (int(round(x)), int(round(y)))
            color = (0, 255, 0) if int(lab) == 1 else (255, 0, 0)  # RGB
            cv2.circle(rgb, pt, r, color, thickness=-1)
            cv2.circle(alpha, pt, r, 255, thickness=-1)
    if draw_bbox:
        x1, y1, x2, y2 = bbox_xyxy_px.tolist()
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        thickness = int(max(1, bbox_thickness))
        cv2.rectangle(
            rgb,
            pt1,
            pt2,
            (0, 255, 0),  # RGB green
            thickness=thickness,
        )
        # Also make the bbox visible in the RGBA output even when alpha=0 by forcing alpha=255 on the box lines.
        cv2.rectangle(alpha, pt1, pt2, 255, thickness=thickness)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)

    output_rgba_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(output_rgba_path)
    print(f"[OK] {image_path.name} -> {output_rgba_path.name} (score={chosen_score:.4f})")

# --- 主程序入口 ---
if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.draw_bbox = True
        args.draw_points = True

    candidate_views_dir = Path(args.candidate_views_dir)
    images_dir = candidate_views_dir / args.images_subdir
    masks_dir = candidate_views_dir / args.mask_subdir
    out_dir = candidate_views_dir / args.output_subdir

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Missing mask dir: {masks_dir}")

    # Always overwrite the output directory per run.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor, _, bpe_path, ckpt_path = init_sam3_model(
        bpe_path=args.bpe_path,
        sam3_weights_dir=args.sam3_weights_dir,
        sam3_ckpt_path=args.sam3_ckpt_path,
    )
    print(f"[SAM3] bpe_path: {bpe_path}")
    if ckpt_path:
        print(f"[SAM3] ckpt_path: {ckpt_path}")
    else:
        print("[SAM3] ckpt_path: <none> (using library defaults)")

    image_paths = sorted(images_dir.glob(args.pattern))
    if args.instance_id is not None:
        image_paths = [p for p in image_paths if f"instance_{int(args.instance_id)}_" in p.name]

    if not image_paths:
        print("[Info] No images matched.")
        raise SystemExit(0)

    missing = 0
    for image_path in image_paths:
        stem = image_path.stem  # instance_{id}_rank_01
        output_rgba_path = out_dir / f"{stem}.png"
        coarse_mask_path = masks_dir / f"{stem}_mask.png"
        if not coarse_mask_path.exists():
            print(f"[Warn] Missing coarse mask: {coarse_mask_path}")
            missing += 1
            continue
        refine_from_coarse_mask(
            model=model,
            processor=processor,
            image_path=image_path,
            coarse_mask_path=coarse_mask_path,
            output_rgba_path=output_rgba_path,
            bbox_padding=args.bbox_padding,
            draw_bbox=args.draw_bbox,
            draw_points=args.draw_points,
            bbox_thickness=args.bbox_thickness,
            min_mask_pixels=args.min_mask_pixels,
            box_mode=args.box_mode,
            use_points=args.use_points,
            add_neg_point=args.add_neg_point,
            point_radius=args.point_radius,
            use_mask_prompt=args.use_mask_prompt,
            mask_prompt_strength=args.mask_prompt_strength,
            mask_prompt_gamma=args.mask_prompt_gamma,
            mask_prompt_lowres=args.mask_prompt_lowres,
            iters=args.iters,
            debug=args.debug,
            overwrite=args.overwrite,
        )

    if missing:
        print(f"[Info] Missing masks for {missing} images.")
