#!/usr/bin/env python3
"""Inspect generated objects against the actual calibrated input views."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import trimesh,torch
import nvdiffrast.torch as dr
from inspect_multiview_geometry import render,renderer_check
ROOT=Path(__file__).resolve().parents[2]

def main():
 p=argparse.ArgumentParser();p.add_argument('--ids',type=int,nargs='*');p.add_argument('--branch',default='input');p.add_argument('--output-name',default='reconstruction');args=p.parse_args()
 root=ROOT/'results/scene_gen_room/full_objects';inventory=json.loads((root/'inventory.json').read_text())['objects'];ctx=dr.RasterizeCudaContext();renderer_check(ctx)
 records=[]
 with torch.no_grad():
  for obj in inventory:
   if args.ids and obj['instance_id'] not in args.ids:continue
   directory=root/obj['directory']/args.branch;out=directory/args.output_name
   if not (out/'mesh_world.glb').exists():continue
   existing=out/'inspection.json'
   input_meta=root/obj['directory']/'input/transforms.json'
   if existing.exists() and existing.stat().st_mtime_ns>=max((out/'mesh_world.glb').stat().st_mtime_ns,(directory/'transforms.json').stat().st_mtime_ns,input_meta.stat().st_mtime_ns):
    records.append(json.loads(existing.read_text()));continue
   mesh=trimesh.load(out/'mesh_world.glb',force='mesh',process=False);meta=json.loads((directory/'transforms.json').read_text())
   original=json.loads((root/obj['directory']/'input/transforms.json').read_text());rows=[];stats=[]
   for j,fr in enumerate(meta['frames']):
    K=np.asarray(fr['K_image_pix']);w2c=np.linalg.inv(np.asarray(fr['source_camera']['c2w']));n=256;K[:2]*=.5
    color,pred,depth=render(ctx,mesh.vertices,mesh.faces,K,w2c,n,n)
    observed=Image.open(root/obj['directory']/'input'/original['frames'][j]['file_path']).convert('RGBA').resize((n,n),Image.Resampling.LANCZOS)
    gt=np.asarray(observed)[...,3]>127
    background=Image.new('RGB',(n,n),'white');background.paste(observed,mask=observed.getchannel('A'))
    rows.append((background,Image.fromarray((color*255).astype(np.uint8))))
    stats.append({'source':fr['source_image'],'observed_mask_recall':float((gt&pred).sum()/max(1,gt.sum())),
                  'silhouette_iou':float((gt&pred).sum()/max(1,(gt|pred).sum()))})
   board=Image.new('RGB',(n*2,n*len(rows)+24),'white');ImageDraw.Draw(board).text((5,5),f"{obj['name']}: observed | generated geometry",fill='black')
   for i,row in enumerate(rows):
    for j,im in enumerate(row):board.paste(im,(j*n,24+i*n))
   board.save(out/'inspection.jpg',quality=88)
   record={'instance_id':obj['instance_id'],'category':obj['category'],'faces':len(mesh.faces),'vertices':len(mesh.vertices),'watertight':bool(mesh.is_watertight),
           'mean_observed_mask_recall':float(np.mean([x['observed_mask_recall'] for x in stats])),
           'mean_silhouette_iou':float(np.mean([x['silhouette_iou'] for x in stats])),'views':stats,
           'scope':'Conditioning-view silhouette fit. Occluded completions can increase silhouette area; not ground-truth geometry accuracy.'}
   (out/'inspection.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record);print(obj['name'],round(record['mean_observed_mask_recall'],3),round(record['mean_silhouette_iou'],3),flush=True)
 (root/f'inspection_{args.branch}_{args.output_name}.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
