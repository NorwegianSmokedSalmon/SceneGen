#!/usr/bin/env python3
"""Free API completion of public benchmark cutouts, with calibrated image registration."""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from PIL import Image
import cv2
ROOT=Path(__file__).resolve().parents[1]


def prepare_job(root,obj,j,download):
    directory=root/obj['directory'];meta=json.loads((directory/'input/transforms.json').read_text());fr=meta['frames'][j]
    out=directory/'api';out.mkdir(exist_ok=True)
    filename=fr['source_image']+'.JPG';record=next(x for x in download['files'] if x['path']=='images/'+filename)
    source=ROOT/'data/mipnerf360_indoor/room'/record['path'];sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if sha!=record['sha256']:raise ValueError('Public source provenance mismatch')
    proof={'public_project':'https://jonbarron.info/mipnerf360/','public_archive':download['source'],'archive_path':record['archive_path'],'verified_sha256':sha}
    (out/f'provenance_{j:02d}.json').write_text(json.dumps(proof,indent=2)+'\n')
    image=out/f'input_{j:02d}.png';prompt=out/f'prompt_{j:02d}.txt';completed=out/f'completed_{j:02d}.png'
    if not image.exists():
        im=Image.open(directory/'input'/fr['file_path']).convert('RGBA');bg=Image.new('RGB',im.size,'white');bg.paste(im,mask=im.getchannel('A'));bg.save(image)
    if not prompt.exists():
        category=obj['category']
        detail={'armchair':'深灰色扶手椅。椅背和座垫中的白色大洞原本被玩偶遮挡，不是真实镂空，应补成连续的深灰色布面。',
                'sofa':'灰色三人沙发。沙发的座垫、靠背和扶手被前方桌子、杂物遮挡留下白色缺口，应补成连续完整的沙发布面和基座。',
                'side table':'木质桌面和黑色金属框架的C形边桌。只修复木桌面因移除碗、瓶子而产生的白色缺口，恢复完整平整的木质桌面。保留原本开放的金属支架空间，不要把桌腿之间填成实心，不要增加额外的桌腿。'}[category]
        prompt.write_text('这是一张用于三维重建的固定相机照片，物体是'+detail+'只补全缺失的物体表面。严格保持原相机视角、物体位置、大小、外轮廓及已存在纹理不变。不要旋转、缩放、裁剪或重新设计物体。物体外部保持纯白色，不要添加地面阴影、文字或其他物品。输出同一个固定视角的一件完整物体。\n')
    return directory,j,image,prompt,completed


def run_job(job):
    directory,j,image,prompt,completed=job
    command=[sys.executable,str(ROOT/'scripts/complete_multiview_api.py'),'--provider','modelscope','--image',str(image),'--prompt',str(prompt),'--output',str(completed),'--size','1024x1024','--provider-defaults','--timeout','600']
    with (directory/'api'/f'log_{j:02d}.txt').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    return directory,j,result.returncode


def register(directory):
    meta=json.loads((directory/'input/transforms.json').read_text());out=directory/'completed';out.mkdir(exist_ok=True)
    sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.006);records=[]
    for j,fr in enumerate(meta['frames']):
        original=np.asarray(Image.open(directory/'input'/fr['file_path']).convert('RGBA'));known=original[...,3]>127
        api=directory/'api'/f'completed_{j:02d}.png';entry={'view':j,'accepted':False}
        rgba=original.copy()
        if api.exists():
            panel=np.asarray(Image.open(api).convert('RGB').resize((512,512),Image.Resampling.LANCZOS));reference=original[...,:3].copy();reference[~known]=255
            a,da=sift.detectAndCompute(cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY),cv2.erode(known.astype(np.uint8)*255,np.ones((5,5),np.uint8)))
            b,db=sift.detectAndCompute(cv2.cvtColor(panel,cv2.COLOR_RGB2GRAY),None)
            matches=[] if da is None or db is None else [m for pair in cv2.BFMatcher().knnMatch(db,da,k=2) if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance]
            if len(matches)>=12:
                src=np.float32([b[m.queryIdx].pt for m in matches]);dst=np.float32([a[m.trainIdx].pt for m in matches]);matrix,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=10000)
                if matrix is not None:
                    good=inliers.ravel()>0;rmse=float(np.sqrt(np.mean(np.sum((src[good]@matrix[:,:2].T+matrix[:,2]-dst[good])**2,axis=1))))
                    scale=float(np.linalg.norm(matrix[:,0]));angle=float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0])))
                    aligned=cv2.warpAffine(panel,matrix,(512,512),borderValue=(255,255,255));foreground=aligned.min(2)<238
                    coverage=float((foreground&known).sum()/max(1,known.sum()))
                    accepted=int(good.sum())>=12 and rmse<2.5 and .75<scale<1.3 and abs(angle)<7 and coverage>.9
                    entry.update(accepted=bool(accepted),inliers=int(good.sum()),rmse=rmse,scale=scale,rotation_degrees=angle,coverage=coverage,matrix=matrix.tolist())
                    if accepted:
                        y,x=np.nonzero(known);hull=np.zeros((512,512),np.uint8);cv2.fillConvexPoly(hull,cv2.convexHull(np.column_stack([x,y]).astype(np.int32)),1)
                        generated=foreground&(hull>0);alpha=original[...,3:4]/255.;fill=generated[...,None];combined=alpha+fill*(1-alpha)
                        colors=(original[...,:3]*alpha+aligned*fill*(1-alpha))/np.maximum(combined,1e-8)
                        rgba[generated,:3]=np.round(colors[generated]).clip(0,255).astype(np.uint8);rgba[generated,3]=np.round(combined[generated,0]*255).astype(np.uint8)
                        assert np.array_equal(rgba[original[...,3]==255],original[original[...,3]==255])
                        entry['added_pixels']=int((generated&~known).sum())
        Image.fromarray(rgba).save(out/fr['file_path']);records.append(entry)
    report={'views':records,'accepted_views':sum(x['accepted'] for x in records),'method':'Per-view API inpainting; SIFT similarity registration; original opaque pixels preserved; additions inside observed convex hull. Unknown surfaces remain inferred.'}
    meta['completion']=report;(out/'transforms.json').write_text(json.dumps(meta,indent=2)+'\n');(out/'completion.json').write_text(json.dumps(report,indent=2)+'\n')
    print(directory.name,'COMPLETION',report['accepted_views'],'/',len(records),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',type=int,nargs='+',default=[7,3,22]);p.add_argument('--register-only',action='store_true');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    root=ROOT/'results/scene_gen_room/full_objects';inventory=json.loads((root/'inventory.json').read_text())['objects'];objects=[o for o in inventory if o['instance_id'] in a.ids]
    download=json.loads((ROOT/'data/mipnerf360_indoor/room/download_manifest.json').read_text())
    jobs=[]
    for obj in objects:
        meta=json.loads((root/obj['directory']/'input/transforms.json').read_text())
        jobs += [prepare_job(root,obj,j,download) for j in range(len(meta['frames']))]
    if not a.register_only:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for future in as_completed([pool.submit(run_job,j) for j in jobs]):
                directory,j,code=future.result();print('API',directory.name,j,'exit',code,flush=True)
    for obj in objects:register(root/obj['directory'])

if __name__=='__main__':main()
