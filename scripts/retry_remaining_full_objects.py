#!/usr/bin/env python3
"""After the initial batch, rerun weak/missing small objects with multiview constraints."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];root=ROOT/'results/scene_gen_room/full_objects';primary={0,2,3,4,6,7,11,18,19,20,22,24,39,43}
parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int);args=parser.parse_args()
while args.wait_pid is not None:
    try:os.kill(args.wait_pid,0)
    except ProcessLookupError:break
    time.sleep(10)
subprocess.run([sys.executable,str(ROOT/'examples/segmentation/inspect_full_scene_objects.py'),'--branch','input'],check=True)
objects=json.loads((root/'inventory.json').read_text())['objects'];todo=[];solid={'television','rug','curtain','door','picture frame','book','box','remote control','pillow'}
for obj in objects:
    i=obj['instance_id']
    if i in primary:continue
    out=root/obj['directory']/'input/reconstruction';run=json.loads((out/'run.json').read_text()) if (out/'run.json').exists() else {}
    inspection=json.loads((out/'inspection.json').read_text()) if (out/'inspection.json').exists() else {}
    if run.get('status')!='succeeded' or inspection.get('mean_observed_mask_recall',0)<.8:todo.append(obj)
ids=[o['instance_id'] for o in todo];(root/'guided_required.json').write_text(json.dumps({'instance_ids':sorted(primary|set(ids)),'automatic_retry_ids':ids},indent=2)+'\n')
print('AUTOMATIC MULTIVIEW RETRIES',ids,flush=True)
solid_ids=[o['instance_id'] for o in todo if o['category'] in solid]
if solid_ids:subprocess.run([sys.executable,str(ROOT/'examples/segmentation/prepare_solid_object_inputs.py'),'--ids',*map(str,solid_ids)],check=True)
transforms=[str(root/o['directory']/('amodal' if o['category'] in solid else 'input')/'transforms.json') for o in todo]
if transforms:subprocess.run([sys.executable,str(ROOT/'examples/segmentation/generate_worldsculpt_objects.py'),'--sampler','train','--silhouette-guide','--discard-raw','--output-name','reconstruction_guided_depth','--glb-faces','150000','--continue-on-error','--transforms',*transforms],check=True)
