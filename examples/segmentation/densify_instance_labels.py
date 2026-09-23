#!/usr/bin/env python3
"""Assign fine Gaussians with depth-tested, multi-view foreground mask votes.

Unlike unconstrained nearest-neighbor fill, visible background pixels count
against a foreground assignment. Sparse GausCluster seeds are preserved.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from generate_instance_views import load_gaussians_from_ckpt, render_depth


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--ckpt',type=Path,required=True)
    parser.add_argument('--instances',type=int,nargs='+',required=True)
    parser.add_argument('--min-views',type=int,default=3)
    parser.add_argument('--agreement',type=float,default=.65)
    parser.add_argument('--depth-tolerance',type=float,default=.025)
    parser.add_argument('--view-stride',type=int,default=3)
    args=parser.parse_args()
    cluster=args.data_dir/'cluster_result'
    tracking=torch.load(cluster/'gauscluster_tracking_data.pt',map_location='cpu',weights_only=False)
    views=tracking['view_data'][::args.view_stride]
    device=torch.device('cuda')
    gaussians=load_gaussians_from_ckpt(str(args.ckpt),device)
    means=gaussians[0];n=len(means)
    counts=torch.zeros((len(args.instances),n),device=device,dtype=torch.int16)
    visible_count=torch.zeros(n,device=device,dtype=torch.int16)
    with torch.inference_mode():
        for view in tqdm(views,desc='Depth-tested mask voting'):
            K=view['K'].to(device);c2w=view['camtoworld'].to(device)
            width,height=view['width'],view['height']
            depth=render_depth(gaussians,c2w.cpu().numpy(),K[None],width,height,device)
            camera=(means-c2w[:3,3])@c2w[:3,:3]
            pixels=camera@K.T
            xy=(pixels[:,:2]/pixels[:,2:]).floor().long()
            valid=(camera[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)
            ids=torch.where(valid)[0]
            x,y=xy[ids].unbind(1)
            observed_depth=depth[y,x]
            surface=(observed_depth>0)&((camera[ids,2]-observed_depth).abs()<torch.maximum(observed_depth*args.depth_tolerance,observed_depth.new_tensor(.005)))
            ids,x,y=ids[surface],x[surface],y[surface]
            mask=torch.from_numpy(np.array(Image.open(view['mask_path'])).astype(np.int16)).to(device)
            labels=mask[y,x]
            visible_count[ids]+=1
            for index,ident in enumerate(args.instances): counts[index,ids[labels==ident]]+=1
        hits,winner=counts.max(dim=0)
        valid=(hits>=args.min_views)&(hits.float()>=visible_count.float()*args.agreement)
        original=np.load(cluster/'instance_labels.npy')
        dense=original.copy()
        # Only expand into unassigned fine Gaussians; do not override seed labels.
        valid &= torch.from_numpy(original<0).to(device)
        target=torch.tensor(args.instances,device=device)[winner]
        dense[valid.cpu().numpy()]=target[valid].cpu().numpy()
    np.save(cluster/'instance_labels_dense.npy',dense)
    report=dict(method='depth_tested_multiview_votes',view_count=len(views),view_stride=args.view_stride,
                min_views=args.min_views,agreement=args.agreement,relative_depth_tolerance=args.depth_tolerance,
                instances=[dict(instance_id=i,sparse_points=int((original==i).sum()),dense_points=int((dense==i).sum())) for i in args.instances])
    (cluster/'dense_labels_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
