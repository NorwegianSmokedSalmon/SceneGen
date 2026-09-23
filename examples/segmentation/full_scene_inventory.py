#!/usr/bin/env python3
"""Inventory and track individual room objects from calibrated multi-view SAM masks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[2]
PROMPTS=['armchair','sofa','footstool','side table','piano','piano bench','television',
         'television stand','loudspeaker','subwoofer','bookshelf','potted plant','rug',
         'curtain','door','picture frame','lamp','stuffed animal','pillow','bowl',
         'glass bottle','drinking glass','cup','slipper','book','remote control','bag','box','vase']


def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def segment(args,views):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    directory=args.output/'segmentation';directory.mkdir(parents=True,exist_ok=True)
    settings={'prompts':PROMPTS,'threshold':.45,'min_pixels':120,'stride':args.stride}
    settings_path=directory/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        raise ValueError('Segmentation settings changed; choose a different output')
    save_json(settings_path,settings)
    chosen=list(range(0,len(views),args.stride))
    todo=[i for i in chosen if not (directory/(views[i]['image_name']+'.json')).exists()]
    if not todo:return
    torch.set_num_threads(8)
    model=build_sam3_image_model(checkpoint_path=str(ROOT/'.cache/scene_gen/models/sam3/sam3.pt'),load_from_HF=False,enable_inst_interactivity=False)
    processor=Sam3Processor(model,device='cuda',confidence_threshold=.45)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for number,index in enumerate(todo):
            v=views[index];name=v['image_name'];im=Image.open(v['image_path']).convert('RGB')
            state=processor.set_image(im);found=[]
            for prompt in PROMPTS:
                processor.reset_all_prompts(state);state=processor.set_text_prompt(prompt,state)
                masks=state['masks'].squeeze(1).cpu().numpy().astype(bool)
                scores=state['scores'].float().cpu().numpy()
                for mask,score in zip(masks,scores):
                    area=int(mask.sum())
                    if area<120:continue
                    found.append((float(score),prompt,mask))
            records=[];accepted=[]
            for score,category,mask in sorted(found,key=lambda x:x[0],reverse=True):
                # Suppress synonymous detections, retaining nested objects such as a pillow on a sofa.
                if any((mask&other).sum()/max(1,(mask|other).sum())>.8 for other in accepted):continue
                y,x=np.nonzero(mask)
                records.append({'id':len(records),'category':category,'score':score,'pixels':int(mask.sum()),
                                'bbox':[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]})
                accepted.append(mask)
            packed=np.packbits(np.stack(accepted).reshape(len(accepted),-1),axis=1) if accepted else np.zeros((0,(im.width*im.height+7)//8),np.uint8)
            np.savez_compressed(directory/(name+'.npz'),masks=packed,shape=np.array([im.height,im.width]))
            save_json(directory/(name+'.json'),{'view_index':index,'image_name':name,'objects':records})
            if index%(args.stride*4)==0:
                rgb=np.asarray(im).copy();draw_image=Image.fromarray(rgb);draw=ImageDraw.Draw(draw_image)
                for rec in records:
                    box=rec['bbox'];color=tuple(int(x) for x in np.random.default_rng(rec['id']+19).integers(60,255,3))
                    draw.rectangle(box,outline=color,width=2);draw.text((box[0],box[1]),f"{rec['id']} {rec['category']}",fill='white',stroke_width=1,stroke_fill='black')
                draw_image.save(directory/(name+'_inventory.jpg'),quality=85)
            print(f"SEGMENT {number+1}/{len(todo)} frame={index} {name}: {len(records)} objects",flush=True)
    save_json(args.output/'segmentation_summary.json',{'frames':len(chosen),'prompts':PROMPTS,'status':'complete'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/scene_gen_room/full_objects')
    p.add_argument('--tracking',type=Path,default=ROOT/'data/mipnerf360_indoor/room/cluster_result/gauscluster_tracking_data.pt')
    p.add_argument('--stride',type=int,default=6)
    p.add_argument('--stage',choices=['segment'],default='segment')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    views=torch.load(args.tracking,map_location='cpu',weights_only=False)['view_data']
    segment(args,views)

if __name__=='__main__':main()
