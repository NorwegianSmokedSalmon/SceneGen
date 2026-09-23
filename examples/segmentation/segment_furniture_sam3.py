#!/usr/bin/env python3
"""Text-guided furniture masks for the existing multi-view GausCluster stage."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('.cache/scene_gen/models/sam3/sam3.pt'))
    parser.add_argument('--prompts', nargs='+', default=['chair', 'table', 'footstool'])
    parser.add_argument('--threshold', type=float, default=.55)
    parser.add_argument('--min-pixels', type=int, default=300)
    parser.add_argument('--output-subdir', default='furniture')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    paths = sorted(p for p in (args.data_dir/'images').iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'})[::args.stride]
    if args.limit: paths = paths[:args.limit]
    if not paths: raise ValueError('No images found')
    output = args.data_dir/'sam'/args.output_subdir
    output.mkdir(parents=True,exist_ok=True)
    preview = output/'previews'; preview.mkdir(exist_ok=True)
    model = build_sam3_image_model(checkpoint_path=str(args.checkpoint),load_from_HF=False,enable_inst_interactivity=False)
    processor = Sam3Processor(model,device='cuda',confidence_threshold=args.threshold)
    report = []
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        for index,path in enumerate(tqdm(paths)):
            image = Image.open(path).convert('RGB')
            state = processor.set_image(image)
            candidates = []
            for prompt in args.prompts:
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(prompt,state)
                masks = state['masks'].squeeze(1).cpu().numpy().astype(bool)
                scores = state['scores'].float().cpu().numpy()
                for mask,score in zip(masks,scores):
                    if mask.sum() >= args.min_pixels:
                        candidates.append((float(score),prompt,mask))
            labels = np.zeros((image.height,image.width),dtype=np.uint16)
            objects = []
            accepted = []
            for score,prompt,mask in sorted(candidates,key=lambda c:c[0],reverse=True):
                if any(np.logical_and(mask,other).sum() / max(np.logical_or(mask,other).sum(),1) > .7 for other in accepted):
                    continue
                visible = mask & (labels == 0)
                if visible.sum() < args.min_pixels: continue
                ident = len(objects)+1
                labels[visible] = ident
                accepted.append(mask)
                objects.append(dict(id=ident,prompt=prompt,score=score,pixels=int(visible.sum())))
            Image.fromarray(labels).save(output/f'{path.stem}.png')
            record = dict(image=path.name,objects=objects)
            (output/f'{path.stem}.json').write_text(json.dumps(record,indent=2))
            report.append(record)
            if index % 40 == 0 or len(paths) <= 10:
                rgb = np.array(image).copy()
                colors = np.random.default_rng(42).integers(30,240,(len(objects)+1,3))
                for obj in objects:
                    mask = labels == obj['id']
                    rgb[mask] = (rgb[mask]*.5 + colors[obj['id']]*.5).astype(np.uint8)
                visual = Image.fromarray(rgb); draw = ImageDraw.Draw(visual)
                for obj in objects:
                    yy,xx = np.where(labels==obj['id'])
                    draw.text((int(np.median(xx)),int(np.median(yy))),f"{obj['id']} {obj['prompt']}",fill='white',stroke_width=1,stroke_fill='black')
                visual.save(preview/f'{path.stem}.jpg')
    manifest = dict(checkpoint=str(args.checkpoint.resolve()),prompts=args.prompts,threshold=args.threshold,frames=report)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved masks:',len(report),'instances:',sum(len(r['objects']) for r in report),'->',output)


if __name__ == '__main__': main()
