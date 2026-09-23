import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from gsplat.rendering import rasterization


def generate_optimized_features(
    num_instances: int,
    dim: int = 16,
    steps: int = 2000,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Generate mutually repulsive feature vectors on a unit hypersphere."""
    if num_instances <= 0:
        raise ValueError("num_instances must be > 0.")
    print(f"[Feature] Generating {num_instances} features in {dim}D...")

    features = torch.randn(num_instances, dim, device=device, dtype=torch.float32)
    features = F.normalize(features, p=2, dim=1)
    features.requires_grad_(True)

    optimizer = torch.optim.SGD([features], lr=1.0, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    eye = torch.eye(num_instances, device=device)
    for _ in tqdm(range(steps), desc="Optimizing features"):
        optimizer.zero_grad()
        feat_norm = F.normalize(features, p=2, dim=1)
        gram = torch.mm(feat_norm, feat_norm.t())
        sim = gram * (1.0 - eye)
        loss = torch.logsumexp(sim * 10.0, dim=1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

    final = F.normalize(features, p=2, dim=1).detach()
    max_sim = (final @ final.t() * (1.0 - eye)).max().item()
    print(f"[Feature] Done. Max cosine similarity: {max_sim:.4f}")
    return final


def generate_contrast_palette(num_entries: int, seed: int = 42) -> np.ndarray:
    """Generate an RGB palette with strong contrast; index 0 is black background."""
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


def load_gaussians_from_ckpt(
    ckpt_paths: List[str], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and merge Gaussian parameters from checkpoint files."""
    means, quats, scales, opacities = [], [], [], []
    print(f"[Gaussians] Loading {len(ckpt_paths)} checkpoint(s)...")
    for ckpt_path in ckpt_paths:
        ckpt_path = str(Path(ckpt_path).expanduser())
        payload = torch.load(ckpt_path, map_location=device)
        if "splats" not in payload:
            raise KeyError(f"Checkpoint missing 'splats' key: {ckpt_path}")
        splats = payload["splats"]
        means.append(splats["means"])
        quats.append(F.normalize(splats["quats"], dim=-1))
        scales.append(torch.exp(splats["scales"]))
        opacities.append(torch.sigmoid(splats["opacities"]))
        print(f"  • Loaded {ckpt_path}")

    def _cat(items: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(items, dim=0)

    return _cat(means), _cat(quats), _cat(scales), _cat(opacities)


def prepare_rendering_tensors(
    tracker: Dict, gaussians: Tuple[torch.Tensor, ...], distinct_features: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build opacities and per-point features for rendering."""
    means, _, _, original_opacities = gaussians
    feature_dim = distinct_features.shape[1]

    render_opacities = original_opacities.to(device=means.device, dtype=torch.float32)
    render_features = torch.zeros(
        (means.shape[0], feature_dim), dtype=torch.float32, device=means.device
    )

    total_point_ids_list = tracker.get("total_point_ids_list", [])
    for inst_id, point_ids in enumerate(total_point_ids_list):
        if len(point_ids) == 0:
            continue
        feat_vec = distinct_features[inst_id]
        pidx = torch.as_tensor(point_ids, dtype=torch.long, device=means.device)
        render_opacities[pidx] = original_opacities[pidx]
        render_features[pidx] = feat_vec

    return render_opacities, render_features


def decode_feature_map(
    feat_image: torch.Tensor,
    distinct_feats: torch.Tensor,
    sim_threshold: float,
    alpha_image: torch.Tensor,
    alpha_threshold: float,
    chunk_size: int,
) -> torch.Tensor:
    """Decode (H, W, D) features into instance ids using chunked similarity."""
    h, w, d = feat_image.shape
    feat_flat = feat_image.reshape(-1, d)
    feat_flat = feat_flat / (feat_flat.norm(dim=1, keepdim=True) + 1e-6)

    max_vals = torch.full(
        (feat_flat.shape[0],), -1.0, device=feat_flat.device
    )
    max_ids = torch.zeros(
        (feat_flat.shape[0],), dtype=torch.long, device=feat_flat.device
    )

    num_instances = distinct_feats.shape[0]
    for start in range(0, num_instances, chunk_size):
        end = min(start + chunk_size, num_instances)
        chunk = distinct_feats[start:end]
        sim = feat_flat @ chunk.t()
        vals, idx = sim.max(dim=1)
        better = vals > max_vals
        max_vals[better] = vals[better]
        max_ids[better] = idx[better] + start

    mask = torch.zeros((h, w), dtype=torch.long, device=feat_image.device)
    valid = max_vals >= sim_threshold
    if alpha_threshold > 0:
        valid = valid & (alpha_image.reshape(-1) >= alpha_threshold)
    if torch.any(valid):
        mask.view(-1)[valid] = max_ids[valid] + 1
    return mask


def render_and_save_masks(
    tracker: Dict,
    view_data: List[Dict],
    gaussians: Tuple[torch.Tensor, ...],
    mask_dir: Path,
    vis_dir: Path,
    feature_dim: int = 16,
    steps: int = 2000,
    sim_threshold: float = 0.1,
    alpha_threshold: float = 0.0,
    chunk_size: int = 512,
    device: torch.device = torch.device("cuda"),
):
    mask_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    total_point_ids_list = tracker.get("total_point_ids_list", [])
    num_instances = len(total_point_ids_list)
    if num_instances == 0:
        print("[Render] No instances to render.")
        return

    distinct_feats = generate_optimized_features(
        num_instances=num_instances,
        dim=feature_dim,
        steps=steps,
        device=device,
    )

    palette = generate_contrast_palette(num_instances + 1, seed=42)

    render_opacities, render_features = prepare_rendering_tensors(
        tracker, gaussians, distinct_feats
    )

    means, quats, scales, _ = gaussians
    means = means.to(device=device, dtype=torch.float32)
    quats = quats.to(device=device, dtype=torch.float32)
    scales = scales.to(device=device, dtype=torch.float32)
    render_opacities = render_opacities.to(device=device, dtype=torch.float32)
    render_features = render_features.to(device=device, dtype=torch.float32)

    print(f"[Render] Rendering {len(view_data)} views...")
    for view in tqdm(view_data, desc="Rendering masks"):
        width, height = int(view["width"]), int(view["height"])
        viewmats = torch.linalg.inv(view["camtoworld"].to(device)).unsqueeze(0)
        Ks = view["K"].to(device).unsqueeze(0)

        with torch.no_grad():
            render_colors, render_alphas, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=render_opacities,
                colors=render_features,
                viewmats=viewmats,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=None,
                packed=False,
                render_mode="RGB",
            )

        feat_image = render_colors[0]
        alpha_image = render_alphas[0, ..., 0]

        mask = decode_feature_map(
            feat_image=feat_image,
            distinct_feats=distinct_feats,
            sim_threshold=sim_threshold,
            alpha_image=alpha_image,
            alpha_threshold=alpha_threshold,
            chunk_size=chunk_size,
        )

        mask_np = mask.detach().cpu().numpy().astype(np.uint16)
        img_name = view["image_name"]
        cv2.imwrite(str(mask_dir / f"{img_name}.png"), mask_np)

        vis = np.zeros((height, width, 3), dtype=np.uint8)
        valid = mask_np > 0
        if np.any(valid):
            indices = np.clip(mask_np[valid], 0, palette.shape[0] - 1)
            vis[valid] = palette[indices]
        cv2.imwrite(
            str(vis_dir / f"{img_name}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

    print(f"[Render] Saved masks to {mask_dir}")
    print(f"[Render] Saved visualizations to {vis_dir}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 2D instance masks directly from 3D clusters."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Scene directory that contains cluster_result/gauscluster_tracking_data.pt.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        required=True,
        help="Checkpoint(s) containing Gaussian parameters.",
    )
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sim_threshold", type=float, default=0.8)
    parser.add_argument("--alpha_threshold", type=float, default=0.0)
    parser.add_argument("--chunk_size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script requires CUDA for rasterization.")

    data_dir = Path(args.data_dir)
    tracking_path = discover_tracking_path(data_dir)
    tracking = torch.load(tracking_path, map_location="cpu", weights_only=False)
    if "view_data" not in tracking or "total_point_ids_list" not in tracking:
        raise KeyError("tracking file must include view_data and total_point_ids_list.")

    view_data = tracking["view_data"]
    gaussians = load_gaussians_from_ckpt(args.ckpt, device=device)

    mask_dir = data_dir / "cluster_result" / "projected_sam"
    vis_dir = data_dir / "cluster_result" / "projected_sam_vis"

    render_and_save_masks(
        tracker=tracking,
        view_data=view_data,
        gaussians=gaussians,
        mask_dir=mask_dir,
        vis_dir=vis_dir,
        feature_dim=args.feature_dim,
        steps=args.steps,
        sim_threshold=args.sim_threshold,
        alpha_threshold=args.alpha_threshold,
        chunk_size=args.chunk_size,
        device=device,
    )


if __name__ == "__main__":
    main()
