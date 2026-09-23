#!/usr/bin/env python3
"""Evaluate object meshes on real photographs excluded from their conditioning views.

All source images trained the scene GS. This is an object-conditioning exclusion,
not an independent scene reconstruction benchmark or ground-truth mesh test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import trimesh
import nvdiffrast.torch as dr
from inspect_multiview_geometry import render,renderer_check

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'results/scene_gen_room/multiview_comparison')
    p.add_argument('--count',type=int,default=10)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    cluster=ROOT/'data/mipnerf360_indoor/room/cluster_result'
    cache=args.root/'additional_views';cache.mkdir(parents=True,exist_ok=True)
    tracking=torch.load(cluster/'gauscluster_tracking_data.pt',map_location='cpu',weights_only=False)
    views=tracking['view_data']
    manifest=[]
    for ident in [0,1]:
        meta=json.loads((args.root/'direct'/f'instance_{ident}'/'transforms.json').read_text())
        source_names={f['source_image'] for f in meta['frames']}
        excluded=[i for i,v in enumerate(views) if v['image_name'] in source_names]
        candidates=[]
        for index,v in enumerate(views):
            if any(abs(index-i)<6 for i in excluded):continue
            mask=np.asarray(Image.open(v['mask_path']))==ident
            if mask.sum()<2000:continue
            y,x=np.nonzero(mask)
            if min(x.min(),y.min(),mask.shape[1]-1-x.max(),mask.shape[0]-1-y.max())<3:continue
            direction=v['camtoworld'][:3,3].numpy()-np.asarray(meta['offset'])
            direction/=np.linalg.norm(direction)
            candidates.append((index,int(mask.sum()),direction))
        if not candidates:raise ValueError('No additional real views')
        selected=[max(candidates,key=lambda x:x[1])]
        while len(selected)<min(args.count,len(candidates)):
            available=[c for c in candidates if all(c[0]!=a[0] for a in selected)]
            # Farthest camera direction among eligible real views, deterministic.
            selected.append(min(available,key=lambda c:(max(float(c[2]@a[2]) for a in selected),-c[1])))
        for index,area,direction in selected:
            v=views[index];name=f'instance_{ident}_{v["image_name"]}'
            mask=np.asarray(Image.open(v['mask_path']))==ident
            Image.fromarray((mask*255).astype('uint8')).save(cache/(name+'_mask.png'))
            manifest.append({'instance_id':ident,'image_name':v['image_name'],'tracking_index':index,
                             'K':v['K'].tolist(),'c2w':v['camtoworld'].tolist(),'image_path':v['image_path'],
                             'mask':name+'_mask.png','depth':name+'_depth.npy','mask_pixels':area,
                             'width':v['width'],'height':v['height'],'scale':meta['scale']})
    (cache/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if any(not (cache/f['depth']).exists() for f in manifest):
        from generate_instance_views import load_gaussians_from_ckpt,render_depth
        device=torch.device('cuda')
        gs=load_gaussians_from_ckpt(str(ROOT/'results/scene_gen_room/ckpts/ckpt_6999_rank0.pt'),device)
        for f in manifest:
            if (cache/f['depth']).exists():continue
            K=torch.tensor(f['K'],dtype=torch.float32,device=device)[None]
            depth=render_depth(gs,np.asarray(f['c2w']),K,f['width'],f['height'],device).cpu().numpy()
            np.save(cache/f['depth'],depth)
            print('Prepared',f['instance_id'],f['image_name'],flush=True)
        del gs;torch.cuda.empty_cache()
    if args.prepare_only:return
    ctx=dr.RasterizeCudaContext();renderer_check(ctx)
    reports=[]
    for branch in ['direct','completed']:
        for ident in [0,1]:
            for recon in ['reconstruction','reconstruction_train']:
                directory=args.root/branch/f'instance_{ident}'/recon
                if not (directory/'mesh_world.glb').exists():continue
                mesh=trimesh.load(directory/'mesh_world.glb',force='mesh',process=False)
                metrics=[]
                for f in [v for v in manifest if v['instance_id']==ident]:
                    K=np.asarray(f['K']);w2c=np.linalg.inv(np.asarray(f['c2w']))
                    _,pred,depth=render(ctx,mesh.vertices,mesh.faces,K,w2c,f['width'],f['height'])
                    gt=np.asarray(Image.open(cache/f['mask']))>127
                    reference=np.load(cache/f['depth']);valid=np.isfinite(reference)&(reference>0)
                    visible=pred & (gt | ~valid | (depth<=reference+f['scale']*.025))
                    inter=int((visible&gt).sum());union=int((visible|gt).sum())
                    overlap=gt&pred&valid
                    values=np.abs(depth[overlap]-reference[overlap])/f['scale']
                    metrics.append({'image_name':f['image_name'],'visible_mask_iou':inter/max(1,union),
                                    'observed_mask_recall':inter/max(1,int(gt.sum())),
                                    'normalized_median_depth_error':float(np.median(values)) if len(values) else None})
                report={'branch':branch,'instance_id':ident,'sampler':json.loads((directory/'run.json').read_text())['sampler'],
                        'view_count':len(metrics),'mean_visible_mask_iou':float(np.mean([m['visible_mask_iou'] for m in metrics])),
                        'mean_observed_mask_recall':float(np.mean([m['observed_mask_recall'] for m in metrics])),
                        'scope':'Views excluded from object conditioning by >=6 frame indices; all still belong to scene GS training. Masks/depth are proxy references.',
                        'views':metrics}
                (directory/'additional_views.json').write_text(json.dumps(report,indent=2)+'\n')
                reports.append(report);print(branch,ident,report['sampler'],report['mean_visible_mask_iou'],flush=True)
    (args.root/'additional_view_metrics.json').write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__':main()
