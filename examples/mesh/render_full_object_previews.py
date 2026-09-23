#!/usr/bin/env python3
"""Contact sheet of measured cutouts and each final, independently placed mesh."""
import json,sys
from pathlib import Path
import numpy as np,torch,trimesh
from PIL import Image,ImageDraw
import nvdiffrast.torch as dr
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'examples/segmentation'))
from inspect_multiview_geometry import render,renderer_check
from assemble_full_object_scene import color_vertices

def colored(ctx,mesh,colors,K,w2c,n):
    v=torch.as_tensor(np.asarray(mesh.vertices),dtype=torch.float32,device='cuda');f=torch.as_tensor(np.asarray(mesh.faces),dtype=torch.int32,device='cuda').contiguous();view=torch.as_tensor(w2c,dtype=torch.float32,device='cuda');vc=torch.cat([v,torch.ones_like(v[:,:1])],1)@view.T;near=.001;far=100.
    clip=torch.stack([2*K[0,0]/n*vc[:,0]+(2*K[0,2]/n-1)*vc[:,2],-2*K[1,1]/n*vc[:,1]+(1-2*K[1,2]/n)*vc[:,2],(far+near)/(far-near)*vc[:,2]-2*far*near/(far-near),vc[:,2]],1)
    rast,_=dr.rasterize(ctx,clip[None].contiguous(),f,(n,n));color=dr.interpolate(torch.as_tensor(colors,dtype=torch.float32,device='cuda')[None].contiguous(),rast,f)[0][0].flip(0);idx=rast[0,:,:,3].long().flip(0)-1
    face=vc[f.long(),:3];normal=torch.nn.functional.normalize(torch.cross(face[:,1]-face[:,0],face[:,2]-face[:,0],dim=-1),dim=-1);light=torch.abs(normal[:,2]);shading=.65+.35*light[idx.clamp_min(0)];color*=shading[...,None];color[idx<0]=1;return color.clamp(0,1).cpu().numpy()

def main():
    root=ROOT/'results/scene_gen_room/full_objects';objects=json.loads((root/'inventory.json').read_text())['objects'];selected={x['instance_id']:x for x in json.loads((root/'selected_meshes.json').read_text())['objects']};n=160;tilew=n*3;tileh=n+42;cols=4;rows=(len(objects)+cols-1)//cols;board=Image.new('RGB',(cols*tilew,rows*tileh),'white');draw=ImageDraw.Draw(board);ctx=dr.RasterizeCudaContext();renderer_check(ctx)
    with torch.no_grad():
        for j,obj in enumerate(objects):
            if obj['instance_id'] not in selected:continue
            record=selected[obj['instance_id']];directory=root/obj['directory'];meta=json.loads((directory/'input/transforms.json').read_text());frame=meta['frames'][0];K=np.asarray(frame['K_image_pix']);K[:2]*=n/512;w2c=np.linalg.inv(np.asarray(frame['source_camera']['c2w']));mesh=trimesh.load(root/record['repaired_mesh'],force='mesh',process=False)
            rgba=Image.open(directory/'input'/frame['file_path']).convert('RGBA').resize((n,n));observed=Image.new('RGB',(n,n),'white');observed.paste(rgba,mask=rgba.getchannel('A'));normal,_,_=render(ctx,mesh.vertices,mesh.faces,K,w2c,n,n);color=colored(ctx,mesh,color_vertices(directory,mesh.vertices),K,w2c,n)
            x=j%cols*tilew;y=j//cols*tileh;draw.text((x+4,y+3),f"{obj['instance_id']:02d} {obj['category']} | coverage {record['observed_mask_recall']:.2f}",fill='black');draw.text((x+4,y+19),'observed | repaired normals | observed colors',fill='black')
            for k,im in enumerate([observed,Image.fromarray((normal*255).astype(np.uint8)),Image.fromarray((color*255).astype(np.uint8))]):board.paste(im,(x+k*n,y+42))
    board.save(root/'final_objects_preview.jpg',quality=90);print(root/'final_objects_preview.jpg')
if __name__=='__main__':main()
