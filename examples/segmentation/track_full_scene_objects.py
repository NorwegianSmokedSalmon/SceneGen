#!/usr/bin/env python3
"""Associate SAM detections in calibrated 3D and select complementary real views."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'examples/segmentation'))
from generate_instance_views import compute_heatmap,rank_views_by_strategy_two


def save_json(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')


def masks_for(directory,name):
    z=np.load(directory/(name+'.npz'));h,w=z['shape']
    return np.unpackbits(z['masks'],axis=1)[:,:h*w].reshape(-1,h,w).astype(bool)


def voxels(points,pitch=.012):
    q=np.floor(points/pitch).astype(np.int64)+8192
    return np.unique(q[:,0]+q[:,1]*16384+q[:,2]*268435456)


def compatible(a,b):
    groups=[{'armchair','sofa'},{'footstool','piano bench'},{'loudspeaker','subwoofer'},
            {'glass bottle','vase','drinking glass','cup'},{'television stand','bookshelf'},
            {'book','box'},{'potted plant','vase'}]
    return a==b or any(a in g and b in g for g in groups)


def make_tracks(args,views):
    directory=args.output/'segmentation';detections=[]
    for path in sorted(directory.glob('DSCF*.json')):
        record=json.loads(path.read_text());i=record['view_index'];v=views[i];name=v['image_name']
        depth=np.load(ROOT/'results/scene_gen_room/simulation/cache'/f'{name}.npz')['depth']
        masks=masks_for(directory,name)
        for r in record['objects']:
            mask=masks[r['id']];yy,xx=np.nonzero(mask & (depth>0))
            if len(xx)<50:continue
            step=max(1,len(xx)//15000);yy=yy[::step];xx=xx[::step];z=depth[yy,xx]
            K=v['K'].numpy();pose=v['camtoworld'].numpy()
            pc=np.column_stack([(xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z])
            points=pc@pose[:3,:3].T+pose[:3,3]
            low,high=np.quantile(points,[.02,.98],axis=0)
            keep=np.all((points>=low-.025)&(points<=high+.025),axis=1);points=points[keep]
            if len(points)<50:continue
            center=np.median(points,axis=0)
            detections.append(dict(view_index=i,image_name=name,mask_id=r['id'],category=r['category'],
                                   score=r['score'],pixels=r['pixels'],bbox=r['bbox'],center=center,
                                   bounds=np.array([low,high]),points=points.astype(np.float32),vox=voxels(points)))
    tracks=[]
    for det in sorted(detections,key=lambda d:(d['view_index'],-d['pixels'])):
        candidates=[]
        for ti,t in enumerate(tracks):
            if not compatible(det['category'],t['category']) or det['view_index'] in t['frames']:continue
            # Nearby objects of the same category must share measured surface evidence.
            if np.linalg.norm(det['center']-t['center'])>max(.10,np.linalg.norm(t['bounds'][1]-t['bounds'][0])*.7):continue
            common=len(np.intersect1d(det['vox'],t['vox'],assume_unique=True))
            overlap=common/max(1,min(len(det['vox']),len(t['vox'])))
            if overlap>.12:candidates.append((overlap,ti))
        if candidates:
            _,ti=max(candidates);t=tracks[ti];t['detections'].append(det);t['frames'].add(det['view_index'])
            t['vox']=np.union1d(t['vox'],det['vox']);t['bounds']=np.array([np.minimum(t['bounds'][0],det['bounds'][0]),np.maximum(t['bounds'][1],det['bounds'][1])]);t['center']=t['bounds'].mean(0)
            t['category']=Counter(d['category'] for d in t['detections']).most_common(1)[0][0]
        else:tracks.append({'category':det['category'],'detections':[det],'frames':{det['view_index']},'vox':det['vox'],'bounds':det['bounds'],'center':det['center']})
    # Merge fragmented tracks only with overlapping surface evidence and no same-frame conflict.
    changed=True
    while changed:
        changed=False
        for i in range(len(tracks)):
            a=tracks[i]
            if not a:continue
            for j in range(i+1,len(tracks)):
                b=tracks[j]
                if not b or a['frames']&b['frames'] or not compatible(a['category'],b['category']):continue
                common=len(np.intersect1d(a['vox'],b['vox'],assume_unique=True))
                if common/max(1,min(len(a['vox']),len(b['vox'])))<.22:continue
                a['detections']+=b['detections'];a['frames']|=b['frames'];a['vox']=np.union1d(a['vox'],b['vox'])
                a['bounds']=np.array([np.minimum(a['bounds'][0],b['bounds'][0]),np.maximum(a['bounds'][1],b['bounds'][1])]);a['center']=a['bounds'].mean(0)
                a['category']=Counter(d['category'] for d in a['detections']).most_common(1)[0][0]
                tracks[j]=None;changed=True
    valid=[];excluded=[]
    for t in tracks:
        if not t:continue
        reason=None
        if len(t['frames'])<3:reason='fewer_than_3_independent_views'
        if max(d['pixels'] for d in t['detections'])<250:reason='too_few_resolved_pixels'
        if reason:
            excluded.append({'category':t['category'],'views':len(t['frames']),'reason':reason});continue
        valid.append(t)
    valid.sort(key=lambda t:(-max(d['pixels'] for d in t['detections']),t['category'],float(t['center'][0])))
    (args.output/'objects').mkdir(exist_ok=True)
    records=[]
    for ident,t in enumerate(valid):
        points=np.concatenate([d['points'] for d in t['detections']]);quant=np.floor(points/.004).astype(np.int64)
        _,idx=np.unique(quant,axis=0,return_index=True);points=points[idx]
        if len(points)>30000:points=points[np.linspace(0,len(points)-1,30000).astype(int)]
        low,high=np.quantile(points,[.005,.995],axis=0);center=(low+high)/2;scale=float((high-low).max()*1.15)
        path=args.output/'objects'/f'object_{ident:03d}';path.mkdir(exist_ok=True)
        np.save(path/'observed_points.npy',points)
        dets=[]
        for d in t['detections']:
            dets.append({k:v for k,v in d.items() if k not in ['center','bounds','points','vox']})
        rec={'instance_id':ident,'name':t['category'].replace(' ','_')+f'_{ident:03d}',
             'category':t['category'],'views':len(dets),'bounds_world':[low.tolist(),high.tolist()],
             'offset':center.tolist(),'scale':scale,'detections':dets,'directory':str(path.relative_to(args.output))}
        save_json(path/'inventory.json',rec);records.append(rec)
    save_json(args.output/'inventory.json',{'objects':records,'excluded_detections':excluded,'tracking':'semantic-compatible surface voxel overlap; >=3 views; 0.012 scene-unit voxels'})
    board=Image.new('RGB',(5*220,((len(records)+4)//5)*220),'white');draw=ImageDraw.Draw(board)
    for j,r in enumerate(records):
        d=max(r['detections'],key=lambda d:d['pixels']);v=views[d['view_index']];im=Image.open(v['image_path']).convert('RGB')
        mask=np.any(masks_for(directory,d['image_name'])[d.get('mask_ids',[d['mask_id']])],axis=0)
        rgba=Image.fromarray(np.dstack([np.asarray(im),mask.astype(np.uint8)*255]));crop=rgba.crop(d['bbox']);crop.thumbnail((214,184))
        tile=Image.new('RGB',(220,220),'white');tile.paste(crop,((220-crop.width)//2,30),crop.getchannel('A'))
        ImageDraw.Draw(tile).text((4,4),f"{j}: {r['category']} ({r['views']} views)",fill='black')
        board.paste(tile,((j%5)*220,(j//5)*220))
    board.save(args.output/'inventory_preview.jpg',quality=90)
    print('TRACKED',len(records),'objects',Counter(r['category'] for r in records),flush=True)
    return records


def prepare(args,views,records):
    from scipy import ndimage
    cameras=np.stack([v['camtoworld'].numpy()[:3,3] for v in views]);directory=args.output/'segmentation'
    selection_reports=[]
    for r in records:
        path=args.output/r['directory'];points=np.load(path/'observed_points.npy');center=np.asarray(r['offset'])
        ids=[d['view_index'] for d in r['detections']];areas=[d['pixels'] for d in r['detections']]
        sphere,heat=compute_heatmap(center,cameras,ids,areas,2048,6.,True,'max');tree=cKDTree(sphere)
        candidates=[]
        for d in r['detections']:
            v=views[d['view_index']];pose=v['camtoworld'].numpy();K=v['K'].numpy();w,h=v['width'],v['height']
            camera=(points-pose[:3,3])@pose[:3,:3];uv=camera@K.T
            xy=np.floor(uv[:,:2]/np.maximum(uv[:,2:],1e-6)).astype(int)
            in_image=(camera[:,2]>.01)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
            pid=np.where(in_image)[0];x,y=xy[pid].T;z=camera[pid,2];pixel=y*w+x
            nearest=np.full(w*h,np.inf);np.minimum.at(nearest,pixel,z)
            first=z<=nearest[pixel]+.006
            depth=np.load(ROOT/'results/scene_gen_room/simulation/cache'/f"{d['image_name']}.npz")['depth'][y,x]
            occluded=(depth>0)&(depth<z-np.maximum(.01,z*.02))
            visible=pid[first&~occluded];direction=pose[:3,3]-center;direction/=np.linalg.norm(direction)
            bbox=d['bbox'];clipped=min(bbox[0],bbox[1],w-bbox[2],h-bbox[3])<3
            hscore=float(heat[tree.query(direction)[1]])*(.12 if clipped else 1.)
            occ=float((occluded&first).sum()/max(1,first.sum()));vis=len(visible)/len(points)
            candidates.append({**d,'name':d['image_name'],'heatmap_score':hscore,'occlusion_ratio':occ,
                               'visibility_ratio':vis,'clipped':clipped,'direction':direction.tolist(),'visible_points':visible})
        ranked=rank_views_by_strategy_two(candidates)
        eligible=[c for c in ranked if c['pixels']>=max(120,max(areas)*.10) and not c['clipped']]
        if len(eligible)<3:eligible=ranked
        selected=[eligible[0]];covered=set(selected[0]['visible_points'].tolist())
        while len(selected)<min(args.views,len(eligible)):
            choices=[c for c in eligible if c['view_index'] not in {x['view_index'] for x in selected}]
            def gain(c):
                similarity=max(float(np.asarray(c['direction'])@np.asarray(x['direction'])) for x in selected)
                angle=float(np.arccos(np.clip(similarity,-1,1)))
                novelty=len(set(c['visible_points'].tolist())-covered)/max(1,len(points))
                quality=c['composite_score']/max(eligible[0]['composite_score'],1e-8)
                return (.12+angle)*(.2+.8*quality)+2*novelty
            candidate=max(choices,key=gain);selected.append(candidate);covered.update(candidate['visible_points'].tolist())
        out=path/'input';out.mkdir(exist_ok=True);frames=[]
        sheet=Image.new('RGB',(512*3,512*((len(selected)+2)//3)),'white')
        for j,d in enumerate(selected):
            v=views[d['view_index']];rgb=np.asarray(Image.open(v['image_path']).convert('RGB'));mask=np.any(masks_for(directory,d['image_name'])[d.get('mask_ids',[d['mask_id']])],axis=0)
            y,x=np.nonzero(mask);side=int(np.ceil(max(x.max()-x.min()+1,y.max()-y.min()+1)*1.25))
            x0=int(np.floor((x.min()+x.max()+1-side)/2));y0=int(np.floor((y.min()+y.max()+1-side)/2));box=[x0,y0,x0+side,y0+side]
            crop=Image.fromarray(np.dstack([rgb,mask.astype(np.uint8)*255])).crop(box).resize((512,512),Image.Resampling.LANCZOS)
            name=f'view_{j:02d}.png';crop.save(out/name)
            context=Image.fromarray(rgb).crop(box).resize((512,512),Image.Resampling.LANCZOS);context.save(out/f'context_{j:02d}.png')
            white=Image.new('RGB',(512,512),'white');white.paste(crop,mask=crop.getchannel('A'));sheet.paste(white,((j%3)*512,(j//3)*512))
            affine=np.array([[512/side,0,-x0*512/side],[0,512/side,-y0*512/side],[0,0,1.]])
            K=affine@v['K'].numpy();pose=v['camtoworld'].numpy().copy()@np.diag([1.,-1.,-1.,1.]);pose[:3,3]=(pose[:3,3]-center)/r['scale']
            quality={k:val for k,val in d.items() if k not in ['visible_points']}
            frames.append({'file_path':name,'subsample_idx':j,'is_anchor':j==0,'source_image':d['image_name'],
                           'transform_matrix':pose.tolist(),'K_image_pix':K.tolist(),'image_size_px':512,
                           'crop_bbox':box,'source_camera':{'c2w':v['camtoworld'].tolist(),'K':v['K'].tolist(),'width':v['width'],'height':v['height']},'quality':quality})
        save_json(out/'transforms.json',{'scale':r['scale'],'offset':r['offset'],'R_box':np.eye(3).tolist(),'instance_id':r['instance_id'],
                                      'category':r['category'],'frames':frames,'bounds_source':'calibrated depth-tested multi-view SAM surfaces'})
        sheet.save(path/'selected_views.jpg',quality=90)
        dirs=np.array([c['direction'] for c in selected]);angle=float(np.degrees(np.arccos(np.clip(dirs@dirs.T,-1,1))).max())
        report={'instance_id':r['instance_id'],'category':r['category'],'selected_views':len(selected),'max_angle_degrees':angle,
                'observed_surface_coverage':len(covered)/len(points),
                'score':'Existing H*(1-O)^2*sqrt(V), followed by angular and surface-coverage greedy selection. O/V use depth-tested measured surface points.',
                'frames':[{k:v for k,v in c.items() if k!='visible_points'} for c in candidates]}
        save_json(path/'view_selection.json',report);selection_reports.append({k:v for k,v in report.items() if k!='frames'})
        print('PREPARED',r['name'],len(selected),'views, span',round(angle,1),flush=True)
    save_json(args.output/'view_selection_summary.json',selection_reports)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'results/scene_gen_room/full_objects');p.add_argument('--views',type=int,default=6);p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(8)
    views=torch.load(ROOT/'data/mipnerf360_indoor/room/cluster_result/gauscluster_tracking_data.pt',map_location='cpu',weights_only=False)['view_data']
    records=json.loads((args.output/'inventory.json').read_text())['objects'] if args.prepare_only else make_tracks(args,views)
    prepare(args,views,records)

if __name__=='__main__':main()
