#!/usr/bin/env python3
"""Choose inspected multiview candidates and repair each mesh independently."""
import argparse,json,hashlib,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2]
GUIDED={0,2,3,4,6,7,11,18,19,20,22,24,39,43,45,48,51,54,55,56,57,58,59,61,63,65,67,68,70,71,72}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def choose(root,obj,partial):
    directory=root/obj['directory'];candidates=[]
    for path in directory.glob('*/reconstruction*/inspection.json'):
        run_path=path.parent/'run.json';source=path.parent/'mesh_world.glb'
        if not run_path.exists() or not source.exists():continue
        run=json.loads(run_path.read_text());inspection=json.loads(path.read_text())
        if run['status']!='succeeded':continue
        if inspection['faces']<20:continue
        score=.7*inspection['mean_observed_mask_recall']+.3*inspection['mean_silhouette_iou']
        candidates.append((score,path,inspection,run))
    if obj['instance_id'] in GUIDED:
        preferred=[x for x in candidates if x[1].parent.name=='reconstruction_guided_depth']
        if not preferred:return None
        if obj['instance_id']==7:candidates=preferred
    if not candidates:return None
    score,path,inspection,run=max(candidates,key=lambda x:x[0])
    if obj['instance_id']==24:
        bounded=[x for x in candidates if x[1].parent.parent.name=='spatial_refined' and x[2]['mean_observed_mask_recall']>=inspection['mean_observed_mask_recall']-.04]
        if bounded:score,path,inspection,run=max(bounded,key=lambda x:x[0])
    source=path.parent/'mesh_world.glb'
    # Weak fits are explicitly surfaced; they are never silently called accurate.
    warnings=[]
    if inspection['mean_observed_mask_recall']<.85:warnings.append('Low observed-silhouette coverage; inspect geometry before simulation interaction')
    if 'inferred' in obj.get('bounds_method','').lower():warnings.append('Placement/scale inferred from silhouettes because depth support was insufficient')
    meta=json.loads((path.parent.parent/'transforms.json').read_text())
    return {'instance_id':obj['instance_id'],'source_mesh':str(source.relative_to(root)),'source_sha256':digest(source),
            'branch':path.parent.parent.name,'variant':path.parent.name,'model':run['model'],'sampler':run['sampler'],'conditioning_views':len(meta['frames']),
            'selection_score':score,'observed_mask_recall':inspection['mean_observed_mask_recall'],'silhouette_iou':inspection['mean_silhouette_iou'],
            'selection_reason':'Calibrated silhouette/depth constrained neural generation for a failed sparse shape' if path.parent.name=='reconstruction_guided_depth' else 'Best inspected candidate by observed silhouette coverage and overlap',
            'warnings':warnings,'hidden_geometry_is_inferred':True,'spatial_validity_preference':obj['instance_id']==24 and path.parent.parent.name=='spatial_refined','guide_report':json.loads((path.parent/'shape_guide.json').read_text()) if (path.parent/'shape_guide.json').exists() else None}


def repair(root,obj,record):
    directory=root/obj['directory']/'final';directory.mkdir(exist_ok=True);out=directory/'mesh_world.ply';stamp=directory/'selection.json'
    if stamp.exists() and out.exists():
        previous=json.loads(stamp.read_text())
        if previous.get('source_sha256')==record['source_sha256']:return previous
    repair_file=out.with_suffix('.json')
    if out.exists() and repair_file.exists() and out.stat().st_mtime_ns<=repair_file.stat().st_mtime_ns and out.stat().st_mtime_ns>=(root/record['source_mesh']).stat().st_mtime_ns:
        report=json.loads(repair_file.read_text())
        if report.get('source')==str((root/record['source_mesh']).resolve()) and report['after']['watertight'] and report['source_surface_retention_at_2pct_extent']>=.97:
            record.update(repaired_mesh=str(out.relative_to(root)),repair_report=str(repair_file.relative_to(root)),repair_method=report['repair'],watertight=True,source_surface_retention=report['source_surface_retention_at_2pct_extent'])
            stamp.write_text(json.dumps(record,indent=2)+'\n');return record
    command=[sys.executable,str(ROOT/'examples/mesh/repair_generated_object.py'),'--input',str(root/record['source_mesh']),'--output',str(out),'--max-faces','120000']
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2'}
    with (directory/'repair.log').open('w') as log:
        try:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=150,env=env)
        except subprocess.TimeoutExpired:result=None
        if result is None or result.returncode:
            log.write('\nRetry: voxel-only closure after failed or timed-out MeshFix.\n');log.flush()
            result=subprocess.run(command+['--voxel-only'],stdout=log,stderr=subprocess.STDOUT,timeout=240,env=env)
        if result.returncode:
            log.write('\nRetry: preserve internal surfaces/cavities during closed-shell extraction.\n');log.flush()
            result=subprocess.run(command+['--voxel-only','--preserve-cavities'],stdout=log,stderr=subprocess.STDOUT,timeout=240,env=env)
        if result.returncode:raise RuntimeError(f"Repair failed for {obj['name']}; see {directory/'repair.log'}")
    report=json.loads(out.with_suffix('.json').read_text());record.update(repaired_mesh=str(out.relative_to(root)),repair_report=str(out.with_suffix('.json').relative_to(root)),repair_method=report['repair'],watertight=report['after']['watertight'],source_surface_retention=report['source_surface_retention_at_2pct_extent'])
    stamp.write_text(json.dumps(record,indent=2)+'\n');print('REPAIRED',obj['name'],record['source_surface_retention'],flush=True);return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--workers',type=int,default=2);a=p.parse_args();root=ROOT/'results/scene_gen_room/full_objects';objects=json.loads((root/'inventory.json').read_text())['objects'];todo=[];missing=[]
    if (root/'guided_required.json').exists():
        GUIDED.clear();GUIDED.update(json.loads((root/'guided_required.json').read_text())['instance_ids'])
    for obj in objects:
        record=choose(root,obj,a.partial)
        if record is None:missing.append(obj['instance_id'])
        else:todo.append((obj,record))
    if missing and not a.partial:raise RuntimeError(f'Uninspected/missing final candidates: {missing}')
    records=[];errors=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(repair,root,obj,record):obj for obj,record in todo}
        for future in as_completed(futures):
            try:records.append(future.result())
            except Exception as exc:errors.append({'instance_id':futures[future]['instance_id'],'error':str(exc)});print('REPAIR_ERROR',errors[-1],flush=True)
    records.sort(key=lambda x:x['instance_id']);report={'status':'complete' if not missing and not errors else 'partial','inventory_count':len(objects),'objects':records,'missing':missing,'errors':errors};(root/'selected_meshes.json').write_text(json.dumps(report,indent=2)+'\n');print('SELECTED',len(records),'MISSING',missing,'ERRORS',errors,flush=True)
    if errors or (missing and not a.partial):raise SystemExit(1)
if __name__=='__main__':main()
