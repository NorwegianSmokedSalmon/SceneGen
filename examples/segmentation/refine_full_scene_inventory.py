#!/usr/bin/env python3
"""Apply reviewed instance merges and robust calibrated bounds before generation."""
import json,sys,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
import cv2,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'examples/segmentation'))
from track_full_scene_objects import masks_for,save_json


def main():
    root=ROOT/'results/scene_gen_room/full_objects';path=root/'inventory.json';raw=root/'inventory_raw.json'
    if not raw.exists():shutil.copyfile(path,raw)
    data=json.loads(raw.read_text());objects={r['instance_id']:r for r in data['objects']}
    merges={1:[5],2:[16],9:[17],11:[33],14:[10,25],20:[29,40],30:[23],42:[60]}
    excluded={21:'Integrated sofa back cushion; included in the sofa object',34:'Integrated sofa back cushion; included in the sofa object',
              26:'False rug detection on the wooden floor, confirmed in source views',
              31:'Composite flower/large-plant detection; separate plant 14 and flower vase 56 retained',
              35:'Composite flower/large-plant detection; separate plant 14 and flower vase 56 retained'}
    views=torch.load(ROOT/'data/mipnerf360_indoor/room/cluster_result/gauscluster_tracking_data.pt',map_location='cpu',weights_only=False)['view_data']
    segmentation=root/'segmentation';mask_cache={};review=[];result=[]
    skip=set(excluded)|{v for vs in merges.values() for v in vs}
    for ident,r in objects.items():
        if ident in skip:continue
        members=[ident]+merges.get(ident,[]);dets=defaultdict(list);point_sets=[]
        for mid in members:
            q=root/objects[mid]['directory']/'observed_points.npy';backup=q.with_name('observed_points_raw.npy')
            if not backup.exists():shutil.copyfile(q,backup)
            point_sets.append(np.load(backup))
            for d in objects[mid]['detections']:dets[d['view_index']].append(d)
        points=np.concatenate(point_sets);prepared=[];votes=np.zeros(len(points),int);total=np.zeros(len(points),int);rays=[];origins=[];heights=[]
        for i,entries in dets.items():
            v=views[i];name=v['image_name']
            if name not in mask_cache:mask_cache[name]=masks_for(segmentation,name)
            ids=sorted({d['mask_id'] for d in entries});mask=np.any(mask_cache[name][ids],axis=0);y,x=np.nonzero(mask)
            box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)];pixels=int(mask.sum())
            record={**max(entries,key=lambda x:x['pixels']),'mask_ids':ids,'bbox':box,'pixels':pixels}
            prepared.append(record)
            K=v['K'].numpy();pose=v['camtoworld'].numpy();h,w=mask.shape
            hull=np.zeros(mask.shape,np.uint8);cv2.fillConvexPoly(hull,cv2.convexHull(np.column_stack([x,y]).astype(np.int32)),1)
            hull=cv2.dilate(hull,np.ones((9,9),np.uint8))
            cam=(points-pose[:3,3])@pose[:3,:3];uv=cam@K.T;xy=np.round(uv[:,:2]/np.maximum(uv[:,2:],1e-6)).astype(int)
            inside=(cam[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
            idx=np.where(inside)[0];support=hull[xy[idx,1],xy[idx,0]]>0
            total[idx]+=1;votes[idx[support]]+=1
            clipped=min(box[0],box[1],w-box[2],h-box[3])<3
            if not clipped:
                pixel=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2,1.]);ray=pose[:3,:3]@np.linalg.inv(K)@pixel;ray/=np.linalg.norm(ray)
                rays.append(ray);origins.append(pose[:3,3]);heights.append(max((box[2]-box[0])/K[0,0],(box[3]-box[1])/K[1,1]))
        good=(votes>=3)&(votes>=total*.65);filtered=points[good]
        if len(filtered)>=100:
            low,high=np.quantile(filtered,[.01,.99],axis=0);center=(low+high)/2;scale=float((high-low).max()*1.20)
            method='measured depth points consistent with >=65% of projected amodal silhouette hulls'
        else:
            # Transparent surfaces can have no valid GS depths; triangulate calibrated silhouette centers.
            rays=np.asarray(rays);origins=np.asarray(origins)
            matrices=np.eye(3)[None]-rays[:,:,None]*rays[:,None,:]
            center=np.linalg.lstsq(matrices.sum(0),np.einsum('nij,nj->i',matrices,origins),rcond=None)[0]
            for _ in range(4):
                distances=np.linalg.norm(np.einsum('nij,nj->ni',matrices,center[None]-origins),axis=1)
                weights=1/np.maximum(distances,.01)
                center=np.linalg.lstsq(np.einsum('n,nij->ij',weights,matrices),np.einsum('n,nij,nj->i',weights,matrices,origins),rcond=None)[0]
            scale=float(np.quantile(np.linalg.norm(origins-center,axis=1)*np.asarray(heights),.75)*1.4)
            low=center-scale/2;high=center+scale/2
            filtered=points[np.all((points>=low)&(points<=high),axis=1)]
            if len(filtered)<30:filtered=np.array([center])+np.random.default_rng(42).normal(size=(100,3))*scale*.08
            method='calibrated multi-view silhouette-center triangulation; GS depth unavailable; scale estimated from image extent'
        r.update(detections=sorted(prepared,key=lambda d:d['view_index']),views=len(prepared),offset=center.tolist(),scale=scale,
                 bounds_world=[low.tolist(),high.tolist()],source_track_ids=members,bounds_method=method)
        if r['category']=='vase' and ident==56:r['category']='flower vase';r['name']='flower_vase_056'
        if ident==42:r['category']='plastic bottle';r['name']='plastic_bottle_042'
        if ident==46:r['category']='glass pitcher';r['name']='glass_pitcher_046'
        if ident==52:r['category']='drinking glass';r['name']='drinking_glass_052'
        target=root/r['directory'];np.save(target/'observed_points.npy',filtered.astype(np.float32));save_json(target/'inventory.json',r);result.append(r)
        review.append({'instance_id':ident,'source_tracks':members,'category':r['category'],'views':r['views'],'scale':scale,'points_before':len(points),'points_after':len(filtered),'bounds_method':method})
    save_json(path,{**data,'objects':result,'reviewed_merges':merges,'reviewed_exclusions':excluded})
    save_json(root/'inventory_review.json',{'objects':review,'excluded':excluded,'notes':'Manual visual review plus source-image mask-overlap checks. Bookshelf books remain separate where individually resolved.'})
    print('REVIEWED',len(result),'objects')
    for r in review:print(r['instance_id'],r['category'],r['views'],'views','scale',round(r['scale'],3),'points',r['points_after'])

if __name__=='__main__':main()
