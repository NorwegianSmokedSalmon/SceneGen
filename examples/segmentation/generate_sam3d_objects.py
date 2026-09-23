#!/usr/bin/env python3
"""Generate SAM 3D Gaussians from candidate RGB/masks/depth and replace instances.

Run in scene_gen_3d. Inputs must be produced by the corrected camera exporter.
The source checkpoint is preserved; outputs use its world coordinate system.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('LIDRA_SKIP_INIT', '1')
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ.setdefault('SPARSE_ATTN_BACKEND', 'xformers')

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / '.cache/scene_gen/vendor'
FIELDS = ('means', 'scales', 'quats', 'opacities', 'sh0', 'shN')


def prepare_config(weights, output):
    from omegaconf import OmegaConf
    import yaml
    config = OmegaConf.load(weights / 'pipeline.yaml')
    config.workspace_dir = str(weights)
    config.depth_model = None  # We supply reconstructed camera-space points.
    config.compile_model = False
    config.rendering_engine = 'pytorch3d'
    config.decode_formats = ['gaussian']
    config.slat_decoder_mesh_config_path = None
    config.slat_decoder_mesh_ckpt_path = None
    config.slat_decoder_gs_4_config_path = None
    config.slat_decoder_gs_4_ckpt_path = None
    config_dir = output / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)

    def configure_dino(node):
        if isinstance(node, dict):
            if node.get('_target_', '').endswith('.dino.Dino'):
                # The SAM 3D checkpoint includes the full DINO state dictionary.
                node['repo_or_dir'] = str(VENDOR / 'dinov2')
                node['source'] = 'local'
                node['backbone_kwargs'] = {'pretrained': False}
            for value in node.values():
                configure_dino(value)
        elif isinstance(node, list):
            for value in node:
                configure_dino(value)

    for key in ('ss_generator_config_path', 'slat_generator_config_path'):
        model = yaml.safe_load((weights / config[key]).read_text())
        configure_dino(model)
        dest = config_dir / Path(config[key]).name
        dest.write_text(yaml.safe_dump(model))
        config[key] = str(dest.resolve())
    OmegaConf.save(config, config_dir / 'pipeline.yaml')
    return config


def load_view(views, camera):
    if camera.get('coordinate_system') != 'checkpoint_world_opencv':
        raise ValueError('Regenerate candidate views with the corrected camera exporter first.')
    c2w = np.asarray(camera['c2w'], dtype=np.float32)
    if not np.allclose(np.linalg.det(c2w[:3, :3]), 1, atol=1e-4):
        raise ValueError('Camera must have a proper right-handed rotation.')
    name = camera['file_name']
    rgb = np.array(Image.open(views / 'images' / name).convert('RGB'))
    mask_image = Image.open(views / 'samrefiner' / name)
    if mask_image.mode != 'RGBA':
        raise ValueError('Expected a SAM3 RGBA result with alpha mask.')
    mask = np.array(mask_image)[..., 3] > 0
    depth = np.load(views / 'depths' / Path(name).with_suffix('.npy'))
    if camera.get('render_flip_lr'):
        rgb, mask, depth = [np.fliplr(x).copy() for x in (rgb, mask, depth)]
    if depth.shape != mask.shape or rgb.shape[:2] != mask.shape:
        raise ValueError('RGB, mask and depth dimensions differ.')
    valid = np.isfinite(depth) & (depth > 0)
    mask &= valid
    if mask.sum() < 100:
        raise ValueError(f'Too few valid foreground pixels in {name}.')
    K = np.asarray(camera['K'], dtype=np.float32)
    y, x = np.mgrid[:depth.shape[0], :depth.shape[1]]
    # gsplat evaluates rays through pixel centers.
    rays = np.stack([(x + .5 - K[0, 2]) / K[0, 0],
                     (y + .5 - K[1, 2]) / K[1, 1], np.ones_like(x)], -1)
    points = (rays * depth[..., None] * np.array([-1, -1, 1])).astype(np.float32)
    points[~valid] = np.nan
    rgba = np.dstack([rgb, mask.astype(np.uint8) * 255])
    return rgba, torch.from_numpy(points), mask


def world_splats(result, camera, sh_count):
    from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
    gaussian = result['gaussian'][0]
    xyz = gaussian.get_xyz.float()
    device = xyz.device
    scale = result['scale'].reshape(3).float()
    if not torch.allclose(scale, scale[0].expand(3), atol=1e-5, rtol=1e-4):
        raise ValueError('Nonuniform object pose scale requires covariance decomposition.')
    pose_R = quaternion_to_matrix(result['rotation'].reshape(4).float())
    translation = result['translation'].reshape(3).float()
    c2w = torch.as_tensor(camera['c2w'], device=device, dtype=torch.float32)
    cv_to_p3d = torch.diag(xyz.new_tensor([-1., -1., 1.]))
    means_camera = (xyz * scale) @ pose_R + translation  # PyTorch3D row vectors
    means = (means_camera @ cv_to_p3d) @ c2w[:3, :3].T + c2w[:3, 3]
    world_R = c2w[:3, :3] @ cv_to_p3d @ pose_R.T
    quats = matrix_to_quaternion(world_R @ quaternion_to_matrix(gaussian.get_rotation.float()))
    features = gaussian.get_features.float()
    if features.shape[1:] != (1, 3):
        raise ValueError(f'Expected SH degree zero from SAM 3D, got {features.shape}.')
    splats = dict(means=means, quats=quats,
                  scales=(gaussian.get_scaling.float() * scale[0]).log(),
                  opacities=torch.logit(gaussian.get_opacity.float().flatten().clamp(1e-6, 1-1e-6)),
                  sh0=features, shN=xyz.new_zeros((len(xyz), sh_count, 3)))
    for key, value in splats.items():
        if not torch.isfinite(value).all():
            raise ValueError(f'Non-finite generated {key}.')
    return {key: value.detach().cpu() for key, value in splats.items()}



def fit_instance_bounds(splats, source, labels, ident):
    """Anchor the completion to its measured instance; keep predicted orientation.

    A single global scale and translation match robust center and maximum span.
    This is coarse placement, not ICP or multi-view shape optimization.
    """
    target = source['means'][torch.from_numpy(labels == ident)]
    visible = splats['opacities'].sigmoid() > .1
    points = splats['means'][visible]
    if len(target) < 10 or len(points) < 10:
        raise ValueError('Too few Gaussian centers for instance-bounds placement.')
    quantiles = torch.tensor([.05, .95])
    target_lo, target_hi = torch.quantile(target, quantiles, dim=0)
    object_lo, object_hi = torch.quantile(points, quantiles, dim=0)
    target_span = (target_hi - target_lo).max()
    object_span = (object_hi - object_lo).max()
    if min(target_span, object_span) <= 1e-6:
        raise ValueError('Degenerate instance bounds.')
    factor = target_span / object_span
    offset = (target_lo + target_hi) / 2 - factor * (object_lo + object_hi) / 2
    fitted = dict(splats)
    fitted['means'] = splats['means'] * factor + offset
    fitted['scales'] = splats['scales'] + factor.log()
    return fitted, {'method': 'instance_bounds', 'scale': float(factor),
                    'translation': offset.tolist(), 'quantiles': [.05, .95]}


def render(splats, camera, path):
    from gsplat import rasterization
    p = {key: value.cuda().contiguous() for key, value in splats.items()}
    rgb, alpha, _ = rasterization(
        means=p['means'], quats=p['quats'], scales=p['scales'].exp(),
        opacities=p['opacities'].sigmoid(), colors=torch.cat([p['sh0'], p['shN']], 1),
        sh_degree=int((p['shN'].shape[1] + 1) ** .5) - 1,
        viewmats=torch.linalg.inv(torch.tensor(camera['c2w'], device='cuda')).unsqueeze(0),
        Ks=torch.tensor(camera['K'], device='cuda').unsqueeze(0),
        width=camera['width'], height=camera['height'], packed=False)
    Image.fromarray((rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(path)
    return alpha[0, ..., 0].cpu().numpy()


def save_splats(splats, path):
    from gsplat import export_splats
    torch.save({'splats': splats, 'step': 0}, path.with_suffix('.pt'))
    export_splats(**splats, format='ply', save_to=str(path.with_suffix('.ply')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--views', type=Path, required=True)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', type=Path, help='Per-Gaussian labels; defaults to instance_labels.npy')
    parser.add_argument('--instances', type=int, nargs='+')
    parser.add_argument('--rank', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--placement', choices=['instance_bounds', 'predicted'], default='instance_bounds')
    parser.add_argument('--weights', type=Path, default=ROOT / '.cache/scene_gen/models/sam-3d-objects/checkpoints')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(VENDOR / 'sam-3d-objects'))
    from hydra.utils import instantiate
    cameras = json.loads((args.views / 'images/cameras.json').read_text())
    cameras = [c for c in cameras if c['rank'] == args.rank and
               (args.instances is None or c['instance_id'] in args.instances)]
    if not cameras or (args.instances and set(args.instances) != {c['instance_id'] for c in cameras}):
        raise ValueError('Requested instance/rank is missing from cameras.json.')
    source = torch.load(args.ckpt, map_location='cpu', weights_only=True)['splats']
    labels_path = args.labels or args.views.parent / 'instance_labels.npy'
    labels = np.load(labels_path)
    if labels.shape != (len(source['means']),):
        raise ValueError('Instance labels must match the source checkpoint Gaussian count.')
    config = prepare_config(args.weights.resolve(), args.output)
    pipeline = instantiate(config)
    generated, report = [], []
    with torch.inference_mode():
        for camera in cameras:
            ident = camera['instance_id']
            out_dir = args.output / f'instance_{ident}'
            out_dir.mkdir(exist_ok=True)
            rgba, pointmap, mask = load_view(args.views, camera)
            Image.fromarray(rgba).save(out_dir / 'input.png')
            result = pipeline.run(rgba, seed=args.seed, pointmap=pointmap.cuda(),
                                  with_mesh_postprocess=False, with_texture_baking=False,
                                  with_layout_postprocess=False,
                                  stage1_inference_steps=args.steps, stage2_inference_steps=args.steps,
                                  decode_formats=['gaussian'])
            splats = world_splats(result, camera, source['shN'].shape[1])
            placement = {'method': 'predicted'}
            if args.placement == 'instance_bounds':
                save_splats(splats, out_dir / 'object_predicted')
                splats, placement = fit_instance_bounds(splats, source, labels, ident)
            generated.append(splats)
            save_splats(splats, out_dir / 'object_world')
            result['gaussian'][0].save_ply(str(out_dir / 'object_local.ply'))
            alpha = render(splats, camera, out_dir / 'object_render.png')
            silhouette = alpha > .1
            iou = float(np.logical_and(silhouette, mask).sum() / np.logical_or(silhouette, mask).sum())
            item = dict(instance_id=ident, category=camera.get('category'), camera=camera, seed=args.seed, steps=args.steps,
                        gaussians=len(splats['means']), source_gaussians=int((labels == ident).sum()),
                        mask_iou=iou, placement=placement, rotation=result['rotation'].cpu().tolist(),
                        translation=result['translation'].cpu().tolist(), scale=result['scale'].cpu().tolist())
            (out_dir / 'generation.json').write_text(json.dumps(item, indent=2))
            report.append(item)
            del result
            torch.cuda.empty_cache()
        keep = torch.from_numpy(~np.isin(labels, [c['instance_id'] for c in cameras]))
        combined = {key: torch.cat([source[key][keep]] + [g[key] for g in generated]) for key in FIELDS}
        save_splats(combined, args.output / 'scene_composed')
        for camera in cameras:
            out_dir = args.output / f"instance_{camera['instance_id']}"
            render(source, camera, out_dir / 'scene_before.png')
            render(combined, camera, out_dir / 'scene_after.png')
    manifest = dict(source_checkpoint=str(args.ckpt.resolve()), instance_labels=str(labels_path.resolve()),
                    source_gaussians=len(source['means']), composed_gaussians=len(combined['means']), objects=report)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
