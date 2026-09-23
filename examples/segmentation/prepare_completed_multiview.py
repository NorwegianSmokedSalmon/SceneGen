#!/usr/bin/env python3
"""Extract edited panels, check layout, and preserve all measured foreground pixels.

Only the unknown image regions come from the API. Requires a 3x2 reference
sheet from prepare_multiview_comparison.py and an inspected completion output.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direct-transforms',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--completed',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    meta=json.loads(args.direct_transforms.read_text())
    count=len(meta['frames']);tile=meta['frames'][0]['image_size_px']
    target=(count*tile,2*tile)
    ref=np.asarray(Image.open(args.template).convert('RGB'))
    completed=Image.open(args.completed).convert('RGB')
    if abs(completed.width/completed.height-target[0]/target[1])>.01:
        raise SystemExit('API changed sheet aspect ratio; inspect the result before reconstructing')
    arr=np.asarray(completed.resize(target,Image.Resampling.LANCZOS))
    if ref.shape!=arr.shape:raise ValueError('Reference layout differs from transforms')
    score=float(structural_similarity(ref[:tile],arr[:tile],channel_axis=2,data_range=255))
    args.output.mkdir(parents=True,exist_ok=True)
    report={'top_reference_ssim':score,'protocol':'preserve observed RGB and alpha; API supplies only previously unknown pixels','views':[]}
    output_images=[]
    for j,fr in enumerate(meta['frames']):
        rgb=arr[tile:,j*tile:(j+1)*tile].copy()
        original=np.array(Image.open(args.direct_transforms.parent/fr['file_path']).convert('RGBA'))
        observed=original[...,3]>127
        known=original[...,3]>0
        # Intended only for this dark chair / wood table on a white sheet.
        foreground=rgb.min(axis=2)<235
        components,labels,stats,_=cv2.connectedComponentsWithStats(foreground.astype(np.uint8),8)
        cleaned=np.zeros_like(foreground)
        for label in range(1,components):
            component=labels==label
            if stats[label,cv2.CC_STAT_AREA]>=16 and (component&observed).any():cleaned|=component
        coverage=float((cleaned&observed).sum()/max(1,observed.sum()))
        rgb[known]=original[known,:3]
        alpha=np.where(cleaned|observed,255,0).astype(np.uint8)
        alpha[known]=original[known,3]
        rgba=np.dstack([rgb,alpha])
        if not np.array_equal(rgba[known],original[known]):
            raise ValueError('Observed RGBA pixels changed during completion composition')
        # Record and enforce a basic drift check before calibration is reused.
        added=int((cleaned&~observed).sum())
        report['views'].append({'view':j,'observed_coverage_before_compositing':coverage,
                                'added_foreground_pixels':added,'observed_pixels':int(observed.sum())})
        output_images.append((fr['file_path'],rgba))
    report['passed_basic_layout_checks']=score>=.85 and all(v['observed_coverage_before_compositing']>=.85 for v in report['views'])
    (args.output/'completion_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed_basic_layout_checks']:
        raise SystemExit('API changed the reference layout/visible silhouette; inspect completion_checks.json. No calibrated input exported.')
    for name,rgba in output_images:Image.fromarray(rgba).save(args.output/name)
    meta['completion']={'image':str(args.completed.resolve()),'reference':str(args.template.resolve()),
                        'checks':report,'camera_pose_invariance':'assumed after layout checks; still requires visual inspection'}
    (args.output/'transforms.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
