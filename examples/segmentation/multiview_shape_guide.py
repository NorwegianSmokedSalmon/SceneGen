"""Calibrated silhouette surface prior for failed sparse-structure generation.

This constrains the neural shape decoder with an inferred visual hull. It cannot
recover unseen concavities. The retained measurements are the calibrated masks.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage


def silhouette_occupancy(transforms, resolution=64):
    transforms=Path(transforms);meta=json.loads(transforms.read_text())
    axis=(np.arange(resolution)+.5)/resolution-.5
    points=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
    votes=np.zeros(len(points),np.int16);visible=np.zeros(len(points),np.int16)
    free_votes=np.zeros(len(points),np.int16);depth_votes=np.zeros(len(points),np.int16)
    depth_root=Path(__file__).resolve().parents[2]/'results/scene_gen_room/simulation/cache'
    observed_root=transforms.parent.parent/'input'
    for frame in meta['frames']:
        mask=np.asarray(Image.open(transforms.parent/frame['file_path']).convert('RGBA'))[...,3]>127
        # Tolerate segmentation / voxel quantization without inventing a camera.
        mask=ndimage.binary_dilation(mask,iterations=3)
        c2w=np.asarray(frame['transform_matrix']);cam=(points-c2w[:3,3])@c2w[:3,:3]
        K=np.asarray(frame['K_image_pix']);z=-cam[:,2]
        x=np.rint(K[0,0]*cam[:,0]/np.maximum(z,1e-8)+K[0,2]).astype(int)
        y=np.rint(-K[1,1]*cam[:,1]/np.maximum(z,1e-8)+K[1,2]).astype(int)
        valid=(z>0)&(x>=0)&(x<mask.shape[1])&(y>=0)&(y<mask.shape[0])
        visible+=valid;votes[valid]+=mask[y[valid],x[valid]]
        # Carve only confidently observed foreground free space. Occluder pixels
        # never constrain the hidden object. Depth is a 3DGS proxy, not ground truth.
        depth_file=depth_root/(frame['source_image']+'.npz')
        if depth_file.exists() and not meta.get('disable_depth_guide',False):
            depth=np.load(depth_file)['depth'].astype(np.float32).squeeze()
            crop=Image.fromarray(depth).crop(frame['crop_bbox']).resize((512,512),Image.Resampling.NEAREST)
            depth=np.asarray(crop)/float(meta['scale'])
            known=np.asarray(Image.open(observed_root/frame['file_path']).convert('RGBA'))[...,3]>240
            known=ndimage.binary_erosion(known,iterations=3)&np.isfinite(depth)&(depth>0)
            indices=np.flatnonzero(valid);indices=indices[known[y[indices],x[indices]]]
            depth_votes[indices]+=1
            free_votes[indices]+=(z[indices]<depth[y[indices],x[indices]]-.025)

    # A point must agree in most available views, with at least two observations.
    occupancy=((visible>=2)&(votes>=np.maximum(2,np.ceil(visible*float(meta.get('silhouette_vote_fraction',.8)))))).reshape((resolution,)*3)
    free_space=(free_votes>=np.maximum(1,np.ceil(depth_votes*.67))).reshape((resolution,)*3)
    occupancy &= ~free_space
    occupancy=ndimage.binary_closing(occupancy,iterations=1)
    occupancy=ndimage.binary_fill_holes(occupancy)
    world=points*float(meta['scale'])@np.asarray(meta.get('R_box',np.eye(3))).T+np.asarray(meta['offset'])
    if 'spatial_guide_bounds_world' in meta:
        b=np.asarray(meta['spatial_guide_bounds_world']);occupancy &= np.all((world>=b[0])&(world<=b[1]),axis=1).reshape(occupancy.shape)
    if 'floor_guide_world_plane' in meta:
        plane=np.asarray(meta['floor_guide_world_plane']);occupancy &= (world@plane[:3]+plane[3]>=0).reshape(occupancy.shape)
    return occupancy


def install_guide(pipeline, transforms, output):
    import torch
    meta=json.loads(Path(transforms).read_text())
    occupancy=silhouette_occupancy(transforms)
    shell=occupancy & ~ndimage.binary_erosion(occupancy,iterations=2)
    idx=np.argwhere(shell).astype(np.int32)
    if len(idx)<30:raise ValueError(f'Insufficient calibrated silhouette support: {len(idx)} voxels')
    original=pipeline.sample_sparse_structure
    def guided(cond,*args,**kwargs):
        sampled=original(cond,*args,**kwargs)
        raw=sampled.detach().cpu().numpy()[:,1:]
        # Retain decoder details only within a narrow dilation of the visual hull.
        near=ndimage.binary_dilation(occupancy,iterations=2)
        raw=raw[near[tuple(raw.T)]]
        coords=np.unique(np.concatenate([raw,idx],0),axis=0)
        if len(coords)>49152:raise ValueError(f'Silhouette guide exceeds token budget: {len(coords)}')
        report={'method':'WorldSculpt sparse structure augmented with calibrated multiview silhouette surface voxels with observed-foreground depth free-space carving; neural shape decoder retained',
                'spatial_guide_bounds_world':meta.get('spatial_guide_bounds_world'),'floor_guide_world_plane':meta.get('floor_guide_world_plane'),'depth_guidance_enabled':not meta.get('disable_depth_guide',False),'silhouette_vote_fraction':meta.get('silhouette_vote_fraction',.8),'sampled_voxels':len(sampled),'retained_sampled_voxels':len(raw),'guide_voxels':len(idx),'combined_voxels':len(coords),'resolution':64,
                'limitation':'Visual hull infers hidden surfaces; foreground depth is a 3DGS proxy, not ground-truth measurement; unseen concavities remain uncertain.'}
        if meta.get('disable_depth_guide',False):report['method']='WorldSculpt sparse structure augmented with calibrated multiview silhouette surface voxels; unreliable proxy depth carving disabled; neural shape decoder retained'
        Path(output,'shape_guide.json').write_text(json.dumps(report,indent=2)+'\n')
        print('MULTIVIEW_SHAPE_GUIDE',json.dumps(report),flush=True)
        return torch.as_tensor(np.column_stack([np.zeros(len(coords),np.int32),coords]),dtype=torch.int32,device=pipeline.device)
    pipeline.sample_sparse_structure=guided
    return original
