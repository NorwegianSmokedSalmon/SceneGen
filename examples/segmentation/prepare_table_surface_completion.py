#!/usr/bin/env python3
"""Complete only the wooden tabletop silhouette; retain open support-frame space."""
import json
from pathlib import Path
import numpy as np,cv2
from scipy import ndimage
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
r=ROOT/'results/scene_gen_room/full_objects/objects/object_022';out=r/'surface_completed';out.mkdir(exist_ok=True);meta=json.loads((r/'input/transforms.json').read_text());records=[]
for fr in meta['frames']:
    rgba=np.asarray(Image.open(r/'input'/fr['file_path']).convert('RGBA')).copy();rgb=rgba[...,:3].astype(float);known=rgba[...,3]>127
    wood=known&(rgb[...,0]>rgb[...,1]*1.08)&(rgb[...,1]>rgb[...,2]*1.07)&(rgb[...,0]>45)
    labels,n=ndimage.label(wood);sizes=np.bincount(labels.ravel());sizes[0]=0;wood=labels==sizes.argmax()
    y,x=np.nonzero(wood);hull=np.zeros(known.shape,np.uint8);cv2.fillConvexPoly(hull,cv2.convexHull(np.column_stack([x,y]).astype(np.int32)),1)
    fill=(hull>0)&~known;_,nearest=ndimage.distance_transform_edt(~wood,return_indices=True);rgba[fill,:3]=rgba[nearest[0][fill],nearest[1][fill],:3];rgba[fill,3]=255
    Image.fromarray(rgba).save(out/fr['file_path']);records.append({'file':fr['file_path'],'inferred_top_pixels':int(fill.sum())})
meta['completion']={'method':'Wood tabletop-only convex silhouette completion. Open metal frame unchanged; hidden color inferred from observed tabletop.','views':records};(out/'transforms.json').write_text(json.dumps(meta,indent=2)+'\n');print(records)
