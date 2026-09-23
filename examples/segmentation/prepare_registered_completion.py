#!/usr/bin/env python3
"""Experimental rescue of shifted API panels using observed texture correspondences.

Fits a 2D similarity transform, preserves fully opaque observed pixels, and adds only
API foreground within the convex hull of the observed silhouette. Registration
cannot establish unseen geometry or multiview consistency; inspect before use.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direct-transforms',type=Path,required=True)
    parser.add_argument('--completed',type=Path,required=True)
    parser.add_argument('--panel-boxes',type=Path,required=True,help='JSON list of [x,y,width,height], one box per view')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    meta=json.loads(args.direct_transforms.read_text())
    boxes=json.loads(args.panel_boxes.read_text())
    if len(boxes)!=len(meta['frames']):raise ValueError('One box required per measured view')
    sheet=np.asarray(Image.open(args.completed).convert('RGB'))
    sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.008)
    cv2.setRNGSeed(42)
    report={'method':'SIFT/RANSAC 2D similarity; isotropic crop padding; alpha-compositing preserves observed contributions; additions restricted to observed convex hull',
            'status':'experimental_registration',
            'limitations':'2D alignment does not verify generated surfaces or camera pose; not validated for automatic replacement in the scene',
            'source_image':str(args.completed.resolve()),'panel_boxes':boxes,'views':[]}
    outputs=[]
    for j,(frame,box) in enumerate(zip(meta['frames'],boxes)):
        x,y,w,h=map(int,box)
        if min(x,y)<0 or min(w,h)<=0 or x+w>sheet.shape[1] or y+h>sheet.shape[0]:raise ValueError('Invalid panel box')
        original=np.array(Image.open(args.direct_transforms.parent/frame['file_path']).convert('RGBA'))
        tile=frame['image_size_px'];known=original[...,3]>0;observed=original[...,3]>127
        reference=original[...,:3].copy();reference[~known]=255
        side=max(w,h);padding=np.full((side,side,3),255,np.uint8)
        px,py=(side-w)//2,(side-h)//2;padding[py:py+h,px:px+w]=sheet[y:y+h,x:x+w]
        panel=cv2.resize(padding,(tile,tile),interpolation=cv2.INTER_CUBIC)
        a,da=sift.detectAndCompute(cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY),cv2.erode(observed.astype(np.uint8)*255,np.ones((5,5),np.uint8)))
        b,db=sift.detectAndCompute(cv2.cvtColor(panel,cv2.COLOR_RGB2GRAY),None)
        if da is None or db is None:raise ValueError('Not enough visible texture for registration')
        matches=[m for pair in cv2.BFMatcher().knnMatch(db,da,k=2) if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance]
        if len(matches)<12:raise ValueError(f'View {j}: too few feature correspondences')
        src=np.float32([b[m.queryIdx].pt for m in matches]);dst=np.float32([a[m.trainIdx].pt for m in matches])
        matrix,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=10000)
        if matrix is None:raise ValueError(f'View {j}: registration failed')
        good=inliers.ravel()>0;errors=np.linalg.norm(src@matrix[:,:2].T+matrix[:,2]-dst,axis=1)
        rmse=float(np.sqrt(np.mean(errors[good]**2)))
        span=np.ptp(dst[good],axis=0);ys,xs=np.nonzero(observed);bbox_span=np.array([np.ptp(xs),np.ptp(ys)])
        aligned=cv2.warpAffine(panel,matrix,(tile,tile),borderValue=(255,255,255))
        foreground=aligned.min(axis=2)<235
        coverage=float((foreground&observed).sum()/observed.sum())
        accepted=int(good.sum())>=12 and good.mean()>=.25 and rmse<=2.5 and (span>=.25*bbox_span).all() and coverage>=.92
        hull=np.zeros((tile,tile),np.uint8)
        cv2.fillConvexPoly(hull,cv2.convexHull(np.column_stack([xs,ys]).astype(np.int32)),1)
        generated=foreground&(hull>0)
        added=generated&~known
        # Put inferred pixels behind the original antialiased foreground.
        # Keeping fractional alpha at an internal hole would leave a white ring.
        alpha=original[...,3:4].astype(np.float32)/255
        fill=generated[...,None].astype(np.float32)
        combined=alpha+fill*(1-alpha)
        colors=(original[...,:3]*alpha+aligned*fill*(1-alpha))/np.maximum(combined,1e-8)
        rgba=original.copy()
        rgba[generated,:3]=np.rint(colors[generated]).clip(0,255).astype(np.uint8)
        rgba[generated,3]=np.rint(combined[generated,0]*255).astype(np.uint8)
        opaque=original[...,3]==255
        if not np.array_equal(rgba[opaque],original[opaque]):raise ValueError('Opaque observed pixels changed')
        entry={'view':j,'matches':len(matches),'inliers':int(good.sum()),'inlier_fraction':float(good.mean()),
               'inlier_rmse_px':rmse,'inlier_span_xy_px':span.tolist(),'observed_coverage':coverage,
               'panel_to_observed_similarity':matrix.tolist(),'added_pixels':int(added.sum()),'antialiased_pixels_composited':int((generated&known&~opaque).sum()),'passed_registration_checks':bool(accepted)}
        report['views'].append(entry);outputs.append((frame['file_path'],rgba))
    args.output.mkdir(parents=True,exist_ok=True)
    report['passed_registration_checks']=all(v['passed_registration_checks'] for v in report['views'])
    (args.output/'registration_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed_registration_checks']:raise SystemExit('Registration quality insufficient; no calibrated inputs exported')
    for name,rgba in outputs:Image.fromarray(rgba).save(args.output/name)
    meta['completion']=report
    (args.output/'transforms.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
