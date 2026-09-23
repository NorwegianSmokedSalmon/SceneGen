#!/usr/bin/env python3
"""Validate exported mesh/UsdPhysics assets without claiming simulator execution."""
import argparse,json
from pathlib import Path
import numpy as np
import open3d as o3d
import torch,trimesh
from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils
ROOT=Path(__file__).resolve().parents[2]


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=ROOT/'results/scene_gen_room/simulation');args=p.parse_args();root=args.directory
 manifest=json.loads((root/'manifest.json').read_text());coordinates=json.loads((root/'coordinates.json').read_text());parts=json.loads((root/'assets/parts.json').read_text())
 stage=Usd.Stage.Open(str(root/'scene.usda'));assert stage.GetDefaultPrim().GetPath().pathString=='/World'
 assert UsdGeom.GetStageUpAxis(stage)=='Z' and UsdGeom.GetStageMetersPerUnit(stage)==1.
 meshes={};rayscene=o3d.t.geometry.RaycastingScene();reports=[];id_names={}
 for part in manifest['parts']:
  name=part['part'];m=trimesh.load(root/'assets'/(name.lower()+'_collision.ply'),force='mesh',process=False)
  assert m.is_watertight and m.is_winding_consistent and np.isfinite(m.vertices).all()
  assert (m.area_faces>1e-12).all()
  meshes[name]=m;legacy=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices),o3d.utility.Vector3iVector(m.faces))
  ident=rayscene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy));id_names[int(ident)]=name
  prim=stage.GetPrimAtPath(f'/World/Room/{name}/Collision');assert prim.HasAPI(UsdPhysics.CollisionAPI)
  assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()=='none'
  reports.append({'part':name,'finite':True,'watertight':True,'consistent_winding':True,'degenerate_faces':0,'faces':len(m.faces)})
 floor_bounds=np.array(manifest['floor_bounds_m']);floor=trimesh.creation.box(extents=floor_bounds[1]-floor_bounds[0]);floor.apply_translation(floor_bounds.mean(0))
 f=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(floor.vertices),o3d.utility.Vector3iVector(floor.faces));ident=rayscene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(f));id_names[int(ident)]='FloorSupport'
 views=torch.load(ROOT/'data/mipnerf360_indoor/room/cluster_result/gauscluster_tracking_data.pt',map_location='cpu',weights_only=False)['view_data']
 T=np.array(coordinates['world_to_simulation']);camera=np.stack([v['camtoworld'].numpy()[:3,3] for v in views]);camera=camera@T[:3,:3].T+T[:3,3]
 occupancy=rayscene.compute_occupancy(o3d.core.Tensor(camera.astype(np.float32)),nsamples=3).numpy()
 clearance=rayscene.compute_distance(o3d.core.Tensor(camera.astype(np.float32))).numpy()
 observed=trimesh.load(root/'scene_observed.ply',force='mesh',process=False)
 from scipy.spatial import cKDTree
 floor_points=observed.vertices[np.abs(observed.vertices[:,2])<.05,:2]
 floor_tree=cKDTree(floor_points)
 probes=[]
 for name in ['FloorSupport','Armchair','SideTable','Footstool']:
  b=floor_bounds if name=='FloorSupport' else np.array(parts[name]['bounds_m'])
  xx=np.linspace(b[0,0]+.09,b[1,0]-.09,24);yy=np.linspace(b[0,1]+.09,b[1,1]-.09,16)
  candidates=[]
  for x in xx:
   for y in yy:
    if name=='FloorSupport' and floor_tree.query([x,y])[0]>.1:continue
    offsets=np.array([[0,0],[.04,0],[-.04,0],[0,.04],[0,-.04]])
    origins=np.c_[offsets+np.array([x,y]),np.full(5,3.5)];rays=np.c_[origins,np.tile([0,0,-1],(5,1))]
    hit=rayscene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)));dist=hit['t_hit'].numpy();ids=hit['geometry_ids'].numpy();normals=hit['primitive_normals'].numpy()
    if not np.isfinite(dist).all() or any(id_names.get(int(i))!=name for i in ids):continue
    heights=3.5-dist
    if np.ptp(heights)>.025 or normals[:,2].min()<.8:continue
    if name=='FloorSupport' and abs(heights[0])>.005:continue
    candidates.append((float(np.ptp(heights)),float(x),float(y),float(heights[0])))
  if name=='FloorSupport' and candidates:
   ordered=[min(candidates,key=lambda c:np.linalg.norm(np.array(c[1:3])-np.median(camera[:,:2],axis=0)))]
   while len(ordered)<min(4,len(candidates)):
    remaining=[c for c in candidates if c not in ordered]
    ordered.append(max(remaining,key=lambda c:min(np.linalg.norm(np.array(c[1:3])-np.array(a[1:3])) for a in ordered)))
  else:ordered=sorted(candidates)
  selected=[]
  for spread,x,y,z in ordered:
   if any(np.linalg.norm(np.array([x,y])-np.array(a['xy']))<.4 for a in selected):continue
   selected.append({'target':name,'xy':[x,y],'surface_z':z,'height_spread':spread,'radius_m':.03,'drop_height_m':.25})
   if len(selected)>=(4 if name=='FloorSupport' else 1):break
  probes+=selected
 report={'status':'passed' if not (occupancy>.5).any() and any(p['target']=='FloorSupport' for p in probes) else 'failed','usd_opened':True,'default_prim':'/World','units_m':1,'up_axis':'Z','colliders':reports,
         'recorded_cameras':len(views),'camera_centers_inside_collision_shells':int((occupancy>.5).sum()),
         'minimum_camera_center_clearance_m':float(clearance.min()),'median_camera_center_clearance_m':float(np.median(clearance)),
         'drop_probe_plan':probes,'scope':'OpenUSD and geometric checks only; not Isaac Sim or PhysX execution'}
 (root/'asset_validation.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
 if (occupancy>.5).any():raise SystemExit('Collision shell intrudes into recorded free camera positions')
 if not any(p['target']=='FloorSupport' for p in probes):raise SystemExit('No free floor test locations')

if __name__=='__main__':main()
