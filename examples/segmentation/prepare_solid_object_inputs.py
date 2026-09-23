#!/usr/bin/env python3
"""Close segmentation holes for known solid, approximately convex scene objects.

This is an explicitly inferred silhouette/color prior, not observed hidden data.
Open-frame furniture, plants, cups and bowls are excluded.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import cv2
ROOT=Path(__file__).resolve().parents[2]
SOLID={'television','rug','curtain','door','picture frame','book','box','remote control','pillow'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--ids',type=int,nargs='*');args=p.parse_args()
 root=ROOT/'results/scene_gen_room/full_objects';inventory=json.loads((root/'inventory.json').read_text())['objects'];records=[]
 for obj in inventory:
  if args.ids is not None:
   if obj['instance_id'] not in args.ids:continue
  elif obj['category'] not in SOLID:continue
  directory=root/obj['directory'];meta=json.loads((directory/'input/transforms.json').read_text());out=directory/'amodal';out.mkdir(exist_ok=True);entries=[]
  for fr in meta['frames']:
   original=np.asarray(Image.open(directory/'input'/fr['file_path']).convert('RGBA'));known=original[...,3]>127
   ys,xs=np.nonzero(known);hull=np.zeros(known.shape,np.uint8);cv2.fillConvexPoly(hull,cv2.convexHull(np.column_stack([xs,ys]).astype(np.int32)),1)
   fill=(hull>0)&~known;rgba=original.copy()
   _,nearest=ndimage.distance_transform_edt(~known,return_indices=True)
   rgba[fill,:3]=original[nearest[0][fill],nearest[1][fill],:3];rgba[fill,3]=255
   # Keep fully observed image samples byte-identical.
   opaque=original[...,3]==255;rgba[opaque]=original[opaque]
   Image.fromarray(rgba).save(out/fr['file_path'])
   entries.append({'file':fr['file_path'],'inferred_pixels':int(fill.sum()),'original_opaque_preserved':bool(np.array_equal(rgba[opaque],original[opaque]))})
  meta['completion']={'method':'Category-restricted convex silhouette completion with nearest observed color. Inferred pixels, not measured geometry. No new camera views.','views':entries}
  (out/'transforms.json').write_text(json.dumps(meta,indent=2)+'\n');records.append({'instance_id':obj['instance_id'],'category':obj['category'],'views':len(entries),'inferred_pixels':sum(x['inferred_pixels'] for x in entries)})
 (root/'solid_input_completion.json').write_text(json.dumps(records,indent=2)+'\n');print('Prepared',len(records),'solid-object multiview inputs')
if __name__=='__main__':main()
