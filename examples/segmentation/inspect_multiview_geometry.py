#!/usr/bin/env python3
"""Render generated GLB meshes in measured cameras and report observed-view fit.

Masks and 3DGS depth are reconstruction proxies, not ground-truth 3D geometry.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import trimesh
import nvdiffrast.torch as dr

ROOT=Path(__file__).resolve().parents[2]


def render(ctx,vertices,faces,K,w2c,width,height):
    v=torch.as_tensor(vertices,dtype=torch.float32,device='cuda')
    f=torch.as_tensor(faces,dtype=torch.int32,device='cuda').contiguous()
    view=torch.as_tensor(w2c,dtype=torch.float32,device='cuda')
    vc=torch.cat([v,torch.ones_like(v[:,:1])],1) @ view.T
    # OpenCV camera (+z forward, +y down) -> OpenGL clip space.
    near,far=.001,100.
    clip=torch.stack([2*K[0,0]/width*vc[:,0]+(2*K[0,2]/width-1)*vc[:,2],
                      -2*K[1,1]/height*vc[:,1]+(1-2*K[1,2]/height)*vc[:,2],
                      (far+near)/(far-near)*vc[:,2]-2*far*near/(far-near),vc[:,2]],1)
    rast,_=dr.rasterize(ctx,clip[None].contiguous(),f,(height,width))
    z=dr.interpolate(vc[None,:,2:3].contiguous(),rast,f)[0][0,:,:,0].flip(0)
    index=rast[0,:,:,3].long().flip(0)-1
    tri=vc[f.long(),:3]
    normals=torch.nn.functional.normalize(torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1),dim=-1)
    color=(normals[index.clamp_min(0)]+1)/2
    color[index<0]=1
    return color.cpu().numpy(),(index>=0).cpu().numpy(),z.cpu().numpy()


def renderer_check(ctx):
    # Asymmetric triangle: a vertical flip would place it in the wrong half.
    v=np.array([[-.2,-.5,2.],[.5,-.5,2.],[-.2,-.1,2.]],dtype=np.float32)
    K=np.array([[100.,0,64],[0,100.,64],[0,0,1]])
    _,mask,depth=render(ctx,v,np.array([[0,1,2]]),K,np.eye(4),128,128)
    assert mask[44,59] and not mask[84,59], 'Raster camera convention mismatch'
    assert abs(float(depth[44,59])-2)<1e-4, 'Depth convention mismatch'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transforms',type=Path,nargs='+',required=True)
    p.add_argument('--output-name',default='reconstruction')
    p.add_argument('--views',type=Path,default=ROOT/'data/mipnerf360_indoor/room/cluster_result/candidate_views')
    args=p.parse_args()
    ctx=dr.RasterizeCudaContext();renderer_check(ctx)
    for tj in args.transforms:
        meta=json.loads(tj.read_text());out=tj.parent/args.output_name
        mesh=trimesh.load(out/'mesh_world.glb',force='mesh',process=False)
        if not np.isfinite(mesh.vertices).all():raise ValueError('Nonfinite mesh vertices')
        report={'instance_id':meta['instance_id'],'category':meta['category'],
                'vertices':len(mesh.vertices),'faces':len(mesh.faces),
                'is_watertight':bool(mesh.is_watertight),
                'world_bounds':mesh.bounds.tolist(),'coordinate_units':'normalized scene units, not calibrated meters',
                'metric_scope':'observed input-view mask fit and 3DGS-depth proxy; not held-out or true 3D accuracy',
                'occlusion_tolerance_fraction_of_cube':.025,'views':[]}
        n=512;board=Image.new('RGB',(n*3,n*len(meta['frames'])+32),'white')
        ImageDraw.Draw(board).text((12,8),'Observed masked RGB | Generated mesh normals | Visible reprojection overlay',fill='black')
        for j,fr in enumerate(meta['frames']):
            source=fr['source_camera'];K=np.array(fr['K_image_pix']);w2c=np.linalg.inv(np.array(source['c2w']))
            normals,pred,depth=render(ctx,mesh.vertices,mesh.faces,K,w2c,n,n)
            rgba=np.asarray(Image.open(tj.parent/fr['file_path']).convert('RGBA'))
            # Always evaluate against the observed mask, also for the API branch.
            observed=args.views/'samrefiner'/source['file_name']
            gt_img=Image.open(observed).convert('RGBA').crop(fr['crop_bbox']).resize((n,n),Image.Resampling.LANCZOS)
            gt=np.asarray(gt_img)[...,3]>127
            bg=Image.new('RGB',(n,n),'white');bg.paste(gt_img,mask=gt_img.getchannel('A'))
            reference_depth=np.load(args.views/'depths'/Path(source['file_name']).with_suffix('.npy'))
            d=np.asarray(Image.fromarray(reference_depth.astype(np.float32)).crop(fr['crop_bbox']).resize((n,n),Image.Resampling.NEAREST))
            valid=np.isfinite(d)&(d>0)
            # Original observed object pixels are retained regardless of depth noise.
            visible=pred & (gt | ~valid | (depth<=d+meta['scale']*.025))
            union=(visible|gt).sum();intersection=(visible&gt).sum()
            overlap=gt & pred & valid
            values=np.abs(depth[overlap]-d[overlap])/meta['scale']
            metrics={'source_image':source['source_image'],'visible_mask_iou':float(intersection/max(1,union)),
                     'observed_mask_recall':float(intersection/max(1,gt.sum())),
                     'normalized_median_depth_error':float(np.median(values)) if len(values) else None,
                     'depth_overlap_pixels':int(overlap.sum())}
            report['views'].append(metrics)
            original=Image.open(args.views/'images'/source['file_name']).convert('RGB').crop(fr['crop_bbox']).resize((n,n),Image.Resampling.LANCZOS)
            overlay=np.asarray(original).copy()
            overlay[visible]=(overlay[visible]*.4+np.array([50,180,250])*.6).astype(np.uint8)
            y=32+j*n
            board.paste(bg,(0,y));board.paste(Image.fromarray((normals*255).astype(np.uint8)),(n,y));board.paste(Image.fromarray(overlay),(2*n,y))
            ImageDraw.Draw(board).text((n+8,y+8),f'view {j+1}: visible IoU {metrics["visible_mask_iou"]:.3f}',fill='black')
        report['mean_visible_mask_iou']=float(np.mean([v['visible_mask_iou'] for v in report['views']]))
        board.save(out/'observed_view_comparison.jpg',quality=90)
        (out/'inspection.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)

if __name__=='__main__':main()
