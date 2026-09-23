#!/usr/bin/env python3
"""Geometry-only WorldSculpt on calibrated multi-view inputs; preserve world placement."""
import argparse
import hashlib
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT/'.cache/scene_gen/vendor/WorldSculpt'
os.environ.setdefault('ATTN_BACKEND','sdpa')
os.environ.setdefault('SPARSE_ATTN_BACKEND','sdpa')
os.environ.setdefault('SPARSE_CONV_BACKEND','flex_gemm')
os.environ.setdefault('TORCH_HOME',str(ROOT/'.cache/scene_gen/torch'))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
sys.path.insert(0,str(VENDOR))


def build_geometry(weights):
    import torch
    from pixal3d import models
    from pixal3d.pipelines import samplers
    from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
    import mv_inference_common as mv
    config=json.loads((weights/'Pixal3D/pipeline.json').read_text())['args']
    loaded={}
    for key in ('sparse_structure_decoder','shape_slat_decoder'):
        print('Loading',key,flush=True)
        loaded[key]=models.from_pretrained(str(weights/'Pixal3D'/config['models'][key])).eval()
    aggregators={};features={}
    stages=[('ss','ss_ft64_mv_lora_ibr_texverse','sparse_structure_flow_model'),
            ('shape','shape_ft1024_mv_lora_ibr_texverse_fixedmem05','shape_slat_flow_model_1024')]
    for stage,directory,key in stages:
        print('Loading multi-view',stage,flush=True)
        directory=weights/directory
        cfg=json.loads((directory/'config.json').read_text())
        t=cfg['trainer']['args']
        loaded[key]=mv.build_mv_denoiser(cfg['models']['denoiser'],t['lora_config'],
            str(weights/'Pixal3D'/(config['models'][key]+'.safetensors')),
            str(directory/'ckpts/denoiser_step0015000.pt'),device='cpu',dtype='bfloat16')
        aggregators[stage]=mv.build_aggregator(t['mv_aggregator']['channels'],
            str(directory/'ckpts/mv_aggregator_step0015000.pt'),device='cuda')
        icfg=copy.deepcopy(t['image_cond_model'])
        icfg['args']['model_name']=str(weights/'dinov3')
        features[stage]=mv.build_image_cond_model(icfg,device='cpu')
    kw={}
    for name in ('sparse_structure_sampler','shape_slat_sampler','tex_slat_sampler'):
        c=config[name];kw[name]=getattr(samplers,c['name'])(**c['args']);kw[name+'_params']=c['params']
    pipeline=Pixal3DImageTo3DPipeline(models=loaded,**kw,
        shape_slat_normalization=config['shape_slat_normalization'],
        tex_slat_normalization=config['tex_slat_normalization'],
        image_cond_model_ss=features['ss'],image_cond_model_shape_1024=features['shape'],
        low_vram=True,default_pipeline_type='1024')
    pipeline.to('cuda')
    return pipeline,aggregators


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transforms',type=Path,nargs='+',required=True)
    p.add_argument('--weights',type=Path,default=ROOT/'.cache/scene_gen/models/worldsculpt')
    p.add_argument('--output-name',default='reconstruction')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--sampler',choices=['official','train'],default='official')
    p.add_argument('--glb-faces',type=int,default=200000)
    p.add_argument('--continue-on-error',action='store_true')
    p.add_argument('--discard-raw',action='store_true',help='Keep exported GLB and provenance, discard the intermediate tensor mesh after successful export')
    p.add_argument('--silhouette-guide',action='store_true',help='Augment sparse shape support with calibrated multiview silhouettes')
    args=p.parse_args()
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is required')
    from reconstruct_object import run_instance
    from types import SimpleNamespace
    opts=SimpleNamespace(views='all',anchor=-1,max_views=0,view_select='fps',resolution=1024,
                         seed=args.seed,sampler=args.sampler,max_num_tokens=49152,
                         no_tex=True,no_glb=False,glb_faces=args.glb_faces,vis_ss=False)
    todo=[]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip()
    for tj in args.transforms:
        meta=json.loads(tj.read_text())
        if len(meta['frames'])<2:raise ValueError('This comparison requires genuine multiple input views')
        digest=hashlib.sha256(tj.read_bytes())
        for frame in meta['frames']:
            digest.update((tj.parent/frame['file_path']).read_bytes())
        digest.update((args.weights/'download_manifest.json').read_bytes())
        if args.silhouette_guide:digest.update((Path(__file__).parent/'multiview_shape_guide.py').read_bytes())
        input_sha256=digest.hexdigest()
        out=tj.parent/args.output_name
        if (out/'run.json').exists() and (out/'mesh_world.glb').exists():
            previous=json.loads((out/'run.json').read_text())
            if previous.get('status')=='succeeded' and all(previous.get(k,False if k=='silhouette_guide' else None)==v for k,v in {'seed':args.seed,'sampler':args.sampler,'glb_face_budget':args.glb_faces,'code_commit':commit,'input_sha256':input_sha256,'silhouette_guide':args.silhouette_guide}.items()):
                print('Already complete:',out);continue
        todo.append((tj,out,input_sha256))
    if not todo:return
    pipeline,aggregators=build_geometry(args.weights.resolve())
    for tj,out,input_sha256 in todo:
        out.mkdir(parents=True,exist_ok=True)
        report=dict(status='running',model='AlayaLab/WorldSculpt',code_commit=commit,
                    transforms=str(tj.resolve()),input_sha256=input_sha256,seed=args.seed,sampler=args.sampler,
                    resolution=1024,geometry_only=True,glb_face_budget=args.glb_faces,silhouette_guide=args.silhouette_guide,
                    torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                    weights_manifest=str((args.weights/'download_manifest.json').resolve()),
                    started_utc=datetime.now(timezone.utc).isoformat())
        (out/'run.json').write_text(json.dumps(report,indent=2)+'\n')
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        original_sampler=None
        try:
            if args.silhouette_guide:
                from multiview_shape_guide import install_guide
                original_sampler=install_guide(pipeline,tj,out)
            with torch.no_grad():run_instance(pipeline,aggregators,tj,opts,out_dir=out)
        except Exception as exc:
            import traceback
            report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
            (out/'run.json').write_text(json.dumps(report,indent=2)+'\n')
            print('GENERATION_FAILED',tj,str(exc),flush=True)
            if not args.continue_on_error:raise
            torch.cuda.empty_cache()
            continue
        finally:
            if original_sampler is not None:pipeline.sample_sparse_structure=original_sampler
        if args.discard_raw and (out/'mesh.pt').exists():(out/'mesh.pt').unlink()
        report.update(status='succeeded',raw_tensor_retained=not args.discard_raw,seconds=time.monotonic()-started,
                      peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3)
        (out/'run.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)

if __name__=='__main__':main()
