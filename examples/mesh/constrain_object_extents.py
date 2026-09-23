#!/usr/bin/env python3
"""Remove generated excursions outside observed object bounds / estimated floor.

Closed boolean caps preserve watertightness. Actual calibrated image reprojection
is checked before accepting each change; original generated meshes are retained.
"""
import json,sys,copy,shutil
from pathlib import Path
import numpy as np,trimesh,torch,manifold3d as mf
from PIL import Image
import nvdiffrast.torch as dr
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'examples/segmentation'))
from inspect_multiview_geometry import render,renderer_check
from repair_generated_object import topology

def fit(ctx,mesh,directory):
    meta=json.loads((directory/'input/transforms.json').read_text());values=[]
    for fr in meta['frames']:
        n=192;K=np.asarray(fr['K_image_pix']);K[:2]*=n/512;pose=np.linalg.inv(np.asarray(fr['source_camera']['c2w']));_,mask,_=render(ctx,mesh.vertices,mesh.faces,K,pose,n,n);gt=np.asarray(Image.open(directory/'input'/fr['file_path']).convert('RGBA').resize((n,n),Image.Resampling.LANCZOS))[...,3]>127;values.append(float((gt&mask).sum()/max(1,gt.sum())))
    return float(np.mean(values))

def cut(mesh,planes):
    model=mf.Manifold(mf.Mesh(np.asarray(mesh.vertices,np.float32),np.asarray(mesh.faces,np.uint32)))
    if str(model.status())!='Error.NoError':raise ValueError(f'Invalid boolean input: {model.status()}')
    for normal,offset in planes:model=model.trim_by_plane(normal,float(offset))
    if model.is_empty():raise ValueError('Spatial constraints removed whole object')
    output=model.to_mesh();result=trimesh.Trimesh(np.asarray(output.vert_properties[:,:3]),np.asarray(output.tri_verts),process=False);trimesh.repair.fix_normals(result,multibody=True)
    if not result.is_watertight:raise ValueError('Boolean caps are not watertight')
    return result

def main():
    root=ROOT/'results/scene_gen_room/full_objects';path=root/'selected_meshes.json';backup=root/'selected_meshes_pre_constraints.json'
    selection=json.loads(path.read_text())
    for record in selection['objects']:
        if 'mesh_before_layout_constraints' in record:
            record['repaired_mesh']=record.pop('mesh_before_layout_constraints')
            record.pop('layout_constraints',None);record.pop('observed_mask_recall_after_constraints',None)
    backup.write_text(json.dumps(selection,indent=2)+'\n')
    inventory={o['instance_id']:o for o in json.loads((root/'inventory.json').read_text())['objects']};T=np.asarray(json.loads((ROOT/'results/scene_gen_room/simulation/coordinates.json').read_text())['world_to_simulation']);normal=T[2,:3];length=np.linalg.norm(normal);floor_plane=(normal/length,-T[2,3]/length);ctx=dr.RasterizeCudaContext();renderer_check(ctx);reports=[]
    with torch.no_grad():
      for record in selection['objects']:
        obj=inventory[record['instance_id']];directory=root/obj['directory'];source=root/record['repaired_mesh'];mesh=trimesh.load(source,force='mesh',process=False);b=np.asarray(obj['bounds_world']);extent=b[1]-b[0];margin=np.maximum(extent*.08,max(extent)*.04);low=b[0]-margin;high=b[1]+margin
        if obj['instance_id']==1:
            before=fit(ctx,mesh,directory);candidate=mesh.copy();metric=candidate.vertices@T[:3,:3].T+T[:3,3];z=metric[:,2];metric[:,2]=.002+(z-z.min())/max(np.ptp(z),1e-9)*.01;inv=np.linalg.inv(T);candidate.vertices=metric@inv[:3,:3].T+inv[:3,3];after=fit(ctx,candidate,directory);accepted=after>=before-.04
            report={'instance_id':1,'source_mesh':str(source.relative_to(root)),'bounds_clip':False,'estimated_floor_clip':False,'planar_rug_prior':True,'thickness_m':.01,'height_is_inferred':True,'before_observed_mask_recall':before,'after_observed_mask_recall':after,'accepted':bool(accepted),'scope':'Remove texture-induced geometric relief from the large floor rug using an inferred 1 cm thickness; preserve generated mesh topology and horizontal outline.'}
            if accepted:
                out=directory/'final/mesh_world_constrained.ply';candidate.export(out);report['after']=topology(candidate);record['mesh_before_layout_constraints']=record['repaired_mesh'];record['repaired_mesh']=str(out.relative_to(root));record['layout_constraints']=report;record['observed_mask_recall_after_constraints']=after
            reports.append(report);print(obj['name'],'planar rug',accepted,'fit',round(before,3),'->',round(after,3),flush=True);continue
        metric_z=mesh.vertices@normal+T[2,3];floor_needed=bool(metric_z.min()<-.08);excess=max(float((low-mesh.bounds[0]).max()),float((mesh.bounds[1]-high).max()));bounds_needed=excess>float(extent.max())*.15 and 'inferred' not in obj.get('bounds_method','').lower() and 'triangulation' not in obj.get('bounds_method','').lower()
        if not floor_needed and not bounds_needed:continue
        floor_z=0.;planes=[]
        if bounds_needed:
            for axis in range(3):
                n=np.zeros(3);n[axis]=1;planes.extend([(n,low[axis]),(-n,-high[axis])])
        if floor_needed:planes.append(floor_plane)
        before=fit(ctx,mesh,directory);candidate=cut(mesh,planes);after=fit(ctx,candidate,directory);accepted=after>=before-.04
        if not accepted and floor_needed and bounds_needed:
            candidate=cut(mesh,[floor_plane]);after=fit(ctx,candidate,directory);accepted=after>=before-.04;bounds_needed=False
        if not accepted and floor_needed:
            relaxed_floor=(floor_plane[0],floor_plane[1]-.05/length);candidate=cut(mesh,[relaxed_floor]);after=fit(ctx,candidate,directory);accepted=after>=before-.04;bounds_needed=False;floor_z=-.05
        report={'instance_id':obj['instance_id'],'source_mesh':str(source.relative_to(root)),'bounds_clip':bool(bounds_needed),'estimated_floor_clip':bool(floor_needed),'floor_z_m':floor_z,'original_min_z_m':float(metric_z.min()),'before_observed_mask_recall':before,'after_observed_mask_recall':after,'accepted':bool(accepted),'bounds_padding_world':margin.tolist(),'scope':'Remove unseen generated excursions; floor height and object extents are reconstruction estimates, not surveyed measurements.'}
        if accepted:
            out=directory/'final/mesh_world_constrained.ply';candidate.export(out);report['after']=topology(candidate);report['final_min_z_m']=float((candidate.vertices@normal+T[2,3]).min());record['mesh_before_layout_constraints']=record['repaired_mesh'];record['repaired_mesh']=str(out.relative_to(root));record['layout_constraints']=report;record['observed_mask_recall_after_constraints']=after
        reports.append(report);print(obj['name'],'accepted',accepted,'fit',round(before,3),'->',round(after,3),'floor',floor_needed,'bounds',bounds_needed,flush=True)
    (root/'layout_constraint_report.json').write_text(json.dumps(reports,indent=2)+'\n');path.write_text(json.dumps(selection,indent=2)+'\n')
if __name__=='__main__':main()
