#!/usr/bin/env python3
"""Prepare calibrated, aspect-preserving WorldSculpt direct/API comparison inputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/scene_gen/vendor/WorldSculpt'))
from prepare_crops_video import isolate_object_crop_rgba


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--views', type=Path, default=ROOT/'data/mipnerf360_indoor/room/cluster_result/candidate_views')
    p.add_argument('--ckpt', type=Path, default=ROOT/'results/scene_gen_room/ckpts/ckpt_6999_rank0.pt')
    p.add_argument('--output', type=Path, default=ROOT/'results/scene_gen_room/multiview_comparison')
    p.add_argument('--instances', type=int, nargs='+', default=[0, 1])
    args = p.parse_args()
    cameras = json.loads((args.views/'images/cameras.json').read_text())
    means = torch.load(args.ckpt, map_location='cpu', weights_only=True)['splats']['means'].numpy()
    labels = np.load(args.views.parent/'instance_labels_dense.npy')
    manifest = {'selection': 'existing observed RGB/SAM3 views: area and frame separation, not H/O/V',
                'source_checkpoint': str(args.ckpt), 'instances': []}
    for ident in args.instances:
        cams = sorted([c for c in cameras if c['instance_id'] == ident], key=lambda c:c['rank'])
        pts = means[labels == ident]
        low, high = np.quantile(pts, [.005, .995], axis=0)
        center = (low + high) / 2
        # Same axis-aligned normalization as WorldSculpt's scene preparation.
        size = float((high-low).max() * 1.10)
        out = args.output/'direct'/f'instance_{ident}'
        out.mkdir(parents=True, exist_ok=True)
        ref = args.output/'api'/f'instance_{ident}'
        ref.mkdir(parents=True, exist_ok=True)
        frames = []
        tile = 512
        sheet = Image.new('RGB', (tile * len(cams), tile * 2), 'white')
        for j, c in enumerate(cams):
            if c['coordinate_system'] != 'checkpoint_world_opencv' or c.get('render_flip_lr'):
                raise ValueError('Expected calibrated, unflipped OpenCV cameras')
            rgba = np.asarray(Image.open(args.views/'samrefiner'/c['file_name']).convert('RGBA'))
            rgb = np.asarray(Image.open(args.views/'images'/c['file_name']).convert('RGB'))
            ys,xs = np.nonzero(rgba[...,3] > 127)
            # Use a square crop with a margin for completion, never stretch the input.
            side = int(np.ceil(max(xs.max()-xs.min()+1, ys.max()-ys.min()+1) * 1.25))
            x0 = int(np.floor((xs.max()+xs.min()+1-side)/2))
            y0 = int(np.floor((ys.max()+ys.min()+1-side)/2))
            box = (x0,y0,x0+side,y0+side)
            crop = Image.fromarray(isolate_object_crop_rgba(rgb,rgba[...,3],box)).resize((tile,tile),Image.Resampling.LANCZOS)
            name = f'view_{j:02d}.png'
            crop.save(out/name)
            context = Image.fromarray(isolate_object_crop_rgba(rgb,np.full(rgb.shape[:2],255,np.uint8),box)).resize((tile,tile),Image.Resampling.LANCZOS)
            context_bg = Image.new('RGB',(tile,tile),'white');context_bg.paste(context,mask=context.getchannel('A'))
            masked_bg = Image.new('RGB',(tile,tile),'white');masked_bg.paste(crop,mask=crop.getchannel('A'))
            sheet.paste(context_bg,(tile*j,0));sheet.paste(masked_bg,(tile*j,tile))
            K = np.asarray(c['K'], dtype=float)
            crop_affine = np.array([[tile/side,0,-x0*tile/side],[0,tile/side,-y0*tile/side],[0,0,1.]])
            Ki = crop_affine @ K
            c2w = np.asarray(c['c2w'],dtype=float) @ np.diag([1.,-1.,-1.,1.])
            c2w[:3,3] = (c2w[:3,3]-center)/size
            # Verify canonicalization/cropping preserves the original projections.
            test_pts = pts[::max(1,len(pts)//1000)]
            oldcam = np.c_[test_pts,np.ones(len(test_pts))] @ np.asarray(c['w2c']).T
            newcam = np.c_[(test_pts-center)/size,np.ones(len(test_pts))] @ np.linalg.inv(c2w @ np.diag([1.,-1.,-1.,1.])).T
            uv1 = oldcam[:,:3] @ K.T;uv1=uv1/uv1[:,2:];uv1=uv1 @ crop_affine.T
            uv2 = newcam[:,:3] @ Ki.T;uv2=uv2/uv2[:,2:]
            error = float(np.max(np.abs(uv1-uv2)))
            if error > .01: raise ValueError(f'Camera roundtrip error {error}')
            frames.append(dict(file_path=name,subsample_idx=j,is_anchor=j==0,
                               source_image=c['source_image'],source_file=c['file_name'],
                               transform_matrix=c2w.tolist(),K_image_pix=Ki.tolist(),image_size_px=tile,
                               crop_bbox=list(box),source_camera=c,projection_error_px=error))
        transforms = dict(scale=size,offset=center.tolist(),R_box=np.eye(3).tolist(),
                          instance_id=ident,category=cams[0]['category'],frames=frames,
                          bounds_source='dense 3DGS labels, 0.5/99.5 percentile with 10% margin')
        (out/'transforms.json').write_text(json.dumps(transforms,indent=2)+'\n')
        sheet.save(ref/'input_template.png')
        prompt = (
            f'Edit this exact 3-column, 2-row reference sheet of the SAME {cams[0]["category"]}. '
            'Keep all six panels in exactly the same positions and sizes. The top row contains real room photographs; '
            'leave the entire top row unchanged. The bottom row contains three matching segmented views on pure white. '
            'ONLY edit the three bottom panels: reconstruct the missing, occluded parts of this one object, remove '
            'the occluding objects, and fill the white holes inside the object with its physically plausible original '
            'surfaces. Use all three top photographs and all three cutouts jointly to preserve the identical design, '
            'dimensions, material, legs, seat and back across views. Keep each bottom object exactly at its original '
            'pixel position, scale, perspective, and camera angle. Do not rotate, zoom, recenter, add labels, rearrange '
            'panels, or redesign the object. Preserve every already-visible object surface. Background stays pure white. '
            'Output only the edited sheet with exactly the original 3:2 aspect ratio.'
        )
        (ref/'prompt.txt').write_text(prompt+'\n')
        manifest['instances'].append(dict(instance_id=ident,category=cams[0]['category'],views=len(cams),
                                         direct=str(out),api=str(ref),scale=size,offset=center.tolist()))
        print(f'instance {ident}: {len(cams)} views, cube={size:.4f}, max projection error={max(f["projection_error_px"] for f in frames):.6f}px')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
