#!/usr/bin/env python3
"""Retain object parts hidden in other cameras instead of shrinking to their overlap."""
import argparse,json,sys,copy
from pathlib import Path
import numpy as np,torch,cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'examples/segmentation'))
from track_full_scene_objects import masks_for

def main():
 p=argparse.ArgumentParser();p.add_argument('--ids',type=int,nargs='+',default=[0,4,6,11,20,24]);p.add_argument('--write',action='store_true');a=p.parse_args();root=ROOT/'results/scene_gen_room/full_objects';data=json.loads((root/'inventory.json').read_text());views=torch.load(ROOT/'data/mipnerf360_indoor/room/cluster_result/gauscluster_tracking_data.pt',map_location='cpu',weights_only=False)['view_data'];reports=[]
 for obj in data['objects']:
  if obj['instance_id'] not in a.ids:continue
  directory=root/obj['directory'];points=np.concatenate([np.load(root/f'objects/object_{i:03d}/observed_points_raw.npy') for i in obj.get('source_track_ids',[obj['instance_id']])]);votes=np.zeros(len(points),int);visible=np.zeros(len(points),int)
  for det in obj['detections']:
   view=views[det['view_index']];mask=np.any(masks_for(root/'segmentation',view['image_name'])[det['mask_ids']],axis=0);y,x=np.nonzero(mask);hull=np.zeros(mask.shape,np.uint8);cv2.fillConvexPoly(hull,cv2.convexHull(np.column_stack([x,y]).astype(np.int32)),1);hull=cv2.dilate(hull,np.ones((7,7),np.uint8));K=view['K'].numpy();pose=view['camtoworld'].numpy();cam=(points-pose[:3,3])@pose[:3,:3];projection=cam@K.T;xy=np.rint(projection[:,:2]/np.maximum(projection[:,2:3],1e-8)).astype(int);inside=(cam[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]>=0)&(xy[:,1]<mask.shape[0]);idx=np.flatnonzero(inside);depth=np.load(ROOT/'results/scene_gen_room/simulation/cache'/f"{det['image_name']}.npz")['depth'].squeeze();d=depth[xy[idx,1],xy[idx,0]];support=hull[xy[idx,1],xy[idx,0]]>0
   # A closer occluder is unknown evidence about this object's shape.
   unoccluded=(~np.isfinite(d))|(d<=0)|(cam[idx,2]<=d+obj['scale']*.02)|support
   visible[idx[unoccluded]]+=1;votes[idx[support]]+=1
  keep=(votes>=np.minimum(2,visible))&(votes>=visible*.65)&(votes>0);filtered=points[keep];assert len(filtered)>100
  low,high=np.quantile(filtered,[.001,.999],axis=0);center=(low+high)/2;scale=float(max(high-low)*1.2);report={'instance_id':obj['instance_id'],'old_bounds':obj['bounds_world'],'new_bounds':[low.tolist(),high.tolist()],'source_points':len(points),'retained_points':len(filtered),'method':'Occlusion-aware multiview silhouette consensus: camera truncation and closer foreground occluders are unknown evidence, not object absence'};reports.append(report);print(json.dumps(report),flush=True)
  if not a.write:continue
  base='amodal' if obj['instance_id'] in {0,4,6} else 'input';meta=json.loads((directory/base/'transforms.json').read_text());meta['offset']=center.tolist();meta['scale']=scale;meta['placement_refinement']=report;out=directory/'bounds_refined';out.mkdir(exist_ok=True)
  for fr in meta['frames']:
   pose=np.asarray(fr['source_camera']['c2w'])@np.diag([1.,-1.,-1.,1.]);pose[:3,3]=(pose[:3,3]-center)/scale;fr['transform_matrix']=pose.tolist();Image.open(directory/base/fr['file_path']).save(out/fr['file_path'])
  (out/'transforms.json').write_text(json.dumps(meta,indent=2)+'\n');obj.update(offset=center.tolist(),scale=scale,bounds_world=[low.tolist(),high.tolist()],bounds_method=report['method']);(directory/'inventory.json').write_text(json.dumps(obj,indent=2)+'\n');backup=directory/'observed_points_before_occlusion_refinement.npy'
  if not backup.exists():np.save(backup,np.load(directory/'observed_points.npy'))
  np.save(directory/'observed_points.npy',filtered.astype(np.float32))
 if a.write:(root/'inventory.json').write_text(json.dumps(data,indent=2)+'\n');(root/'occlusion_bounds_refinement.json').write_text(json.dumps(reports,indent=2)+'\n')
if __name__=='__main__':main()
