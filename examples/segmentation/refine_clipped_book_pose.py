#!/usr/bin/env python3
"""Fix book 74 triangulation: clipped silhouettes do not share the same centroid."""
import json,copy
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];root=ROOT/'results/scene_gen_room/full_objects';directory=root/'objects/object_074';meta=json.loads((directory/'amodal/transforms.json').read_text());original=copy.deepcopy(meta)
anchor=meta['frames'][0];K=np.asarray(anchor['source_camera']['K']);pose=np.asarray(anchor['source_camera']['c2w']);bbox=anchor['quality']['bbox'];uv=np.array([(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2,1.]);ray=pose[:3,:3]@np.linalg.solve(K,uv);base=pose[:3,3];aa=[];bb=[]
for fr in meta['frames'][1:]:
    camera=np.asarray(fr['source_camera']['c2w']);k=np.asarray(fr['source_camera']['K']);q=(base-camera[:3,3])@camera[:3,:3];v=ray@camera[:3,:3];b=fr['quality']['bbox'];u=(b[0]+b[2])/2;delta=(u-k[0,2])/k[0,0];aa.append(v[0]-delta*v[2]);bb.append(q[0]-delta*q[2])
depth=-np.dot(aa,bb)/np.dot(aa,aa);assert depth>0;center=base+ray*depth;scale=(bbox[3]-bbox[1])/K[1,1]*depth*1.25
meta['offset']=center.tolist();meta['scale']=float(scale);meta['disable_depth_guide']=True
method='Clipping-aware triangulation: complete anchor silhouette center ray, horizontal center constraints from two vertically clipped views; height inferred from complete anchor image'
meta['placement_refinement']={'method':method,'previous_offset':original['offset'],'previous_scale':original['scale'],'anchor_depth':float(depth),'depth_is_estimated':True}
out=directory/'pose_refined';out.mkdir(exist_ok=True)
for fr in meta['frames']:
    pose=np.asarray(fr['source_camera']['c2w'])@np.diag([1.,-1.,-1.,1.]);pose[:3,3]=(pose[:3,3]-center)/scale;fr['transform_matrix']=pose.tolist();Image.open(directory/'amodal'/fr['file_path']).save(out/fr['file_path'])
(out/'transforms.json').write_text(json.dumps(meta,indent=2)+'\n')
# Inferred plane samples support color transfer only; never labelled measured depth.
image=np.asarray(Image.open(directory/'input'/anchor['file_path']).convert('RGBA'));y,x=np.nonzero(image[...,3]>127);pix=np.column_stack([x,y,np.ones(len(x))]);cam=pix@np.linalg.inv(np.asarray(anchor['K_image_pix'])).T*depth;pose=np.asarray(anchor['source_camera']['c2w']);points=cam@pose[:3,:3].T+pose[:3,3]

if not (directory/'observed_points_pose_before.npy').exists():np.save(directory/'observed_points_pose_before.npy',np.load(directory/'observed_points.npy'))
np.save(directory/'observed_points.npy',points.astype(np.float32)[::max(1,len(points)//20000)])
inventory=json.loads((root/'inventory.json').read_text());obj=next(o for o in inventory['objects'] if o['instance_id']==74);obj.update(offset=center.tolist(),scale=float(scale),bounds_world=[points.min(0).tolist(),points.max(0).tolist()],bounds_method=method+'; inferred scale and plane depth',placement_refinement=meta['placement_refinement']);(directory/'inventory.json').write_text(json.dumps(obj,indent=2)+'\n');(root/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n');print(meta['placement_refinement'])
