#!/usr/bin/env python3
"""Export sharp observed RGB + clustered SAM3 mask + rendered depth for SAM 3D."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from generate_instance_views import load_gaussians_from_ckpt, render_depth, _depth_to_vis_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--ckpt',type=Path,required=True)
    parser.add_argument('--instances',type=int,nargs='+',required=True)
    parser.add_argument('--topk',type=int,default=3)
    args = parser.parse_args()
    cluster = args.data_dir/'cluster_result'
    tracking = torch.load(cluster/'gauscluster_tracking_data.pt',map_location='cpu',weights_only=False)
    views = tracking['view_data']
    candidates = {ident:[] for ident in args.instances}
    categories = {ident:Counter() for ident in args.instances}
    for index,view in enumerate(views):
        labels = np.array(Image.open(view['mask_path']))
        original_path = args.data_dir/'sam/furniture'/f"{view['image_name']}.png"
        original = np.array(Image.open(original_path))
        metadata = json.loads(original_path.with_suffix('.json').read_text())
        for ident in args.instances:
            mask = labels == ident
            area = int(mask.sum())
            if area < 500: continue
            y,x = np.where(mask)
            clipped = min(x.min(),y.min(),labels.shape[1]-1-x.max(),labels.shape[0]-1-y.max()) < 3
            score = area * (.03 if clipped else 1)
            candidates[ident].append((score,index,area,bool(clipped)))
            for obj in metadata['objects']:
                categories[ident][obj['prompt']] += int(np.logical_and(mask,original==obj['id']).sum())
    device = torch.device('cuda')
    gaussians = load_gaussians_from_ckpt(str(args.ckpt),device)
    output = cluster/'candidate_views'
    for name in ['images','depths','depths_vis','projected_mask','samrefiner']:
        (output/name).mkdir(parents=True,exist_ok=True)
    cameras,objects = [],[]
    for ident in args.instances:
        entries = sorted(candidates[ident],reverse=True)
        chosen=[]
        for item in entries:
            # Avoid near-identical adjacent photographs in the exported top-k.
            if any(abs(item[1]-other[1])<12 for other in chosen): continue
            chosen.append(item)
            if len(chosen)==args.topk: break
        if not chosen: raise ValueError(f'No usable mask for instance {ident}')
        category = categories[ident].most_common(1)[0][0]
        objects.append(dict(instance_id=ident,category=category,semantic_votes=dict(categories[ident]),views=len(entries)))
        for rank,(_,index,area,clipped) in enumerate(chosen,1):
            view=views[index]
            rgb=np.array(Image.open(view['image_path']).convert('RGB'))
            mask=np.array(Image.open(view['mask_path']))==ident
            h,w=rgb.shape[:2]
            if mask.shape!=(h,w):raise ValueError('Mask/RGB dimensions differ')
            K=view['K'].float();c2w=view['camtoworld'].float()
            if not torch.allclose(torch.det(c2w[:3,:3]),torch.tensor(1.),atol=1e-4):raise ValueError('Improper camera')
            name=f'instance_{ident}_rank_{rank:02d}'
            Image.fromarray(rgb).save(output/'images'/f'{name}.png')
            Image.fromarray(mask.astype(np.uint8)*255).save(output/'projected_mask'/f'{name}_mask.png')
            Image.fromarray(np.dstack([rgb,mask.astype(np.uint8)*255])).save(output/'samrefiner'/f'{name}.png')
            depth=render_depth(gaussians,c2w.numpy(),K[None].to(device),w,h,device).cpu().numpy()
            np.save(output/'depths'/f'{name}.npy',depth)
            Image.fromarray(_depth_to_vis_rgb(depth)).save(output/'depths_vis'/f'{name}.png')
            cameras.append(dict(file_name=f'{name}.png',instance_id=ident,rank=rank,width=w,height=h,
                                K=K.tolist(),c2w=c2w.tolist(),w2c=torch.linalg.inv(c2w).tolist(),
                                coordinate_system='checkpoint_world_opencv',render_flip_lr=False,
                                source='observed_rgb_sam3_text_gauscluster',source_image=view['image_name'],
                                category=category,mask_pixels=area,clipped=clipped))
            print(ident,category,rank,view['image_name'],'pixels',area,'clipped',clipped,flush=True)
    (output/'images/cameras.json').write_text(json.dumps(cameras,indent=2)+'\n')
    (output/'selection.json').write_text(json.dumps(objects,indent=2)+'\n')


if __name__=='__main__': main()
