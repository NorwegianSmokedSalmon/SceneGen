#!/usr/bin/env python3
"""Load exported USD in Isaac Sim and run contact/drop tests without saving probes.

Run with Isaac Sim's Python environment. NVIDIA's license must be accepted by
the user before launching; this script does not accept any license agreement.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--directory',type=Path,default=Path(__file__).resolve().parents[1]/'results/scene_gen_room/simulation')
 p.add_argument('--seconds',type=float,default=5.)
 p.add_argument('--furniture-probe',choices=['cube','sphere'],default='cube',help='Use non-rolling cubes to test furniture support; floor probes remain spheres')
 args=p.parse_args();root=args.directory.resolve();sys.argv=[sys.argv[0]]
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':True,'width':960,'height':640,'enable_crashreporter':False,'fast_shutdown':True,
                    'extra_args':['--/app/settings/persistent=false','--/app/asyncRendering=false']})
 report={'status':'running','simulator':'Isaac Sim','scene':str(root/'scene.usda'),'test_seconds':args.seconds}
 try:
  import numpy as np
  import omni.usd,omni.physx
  from pxr import UsdGeom,UsdPhysics,PhysxSchema,PhysicsSchemaTools
  from isaacsim.core.api import World
  from isaacsim.core.api.objects import DynamicSphere,DynamicCuboid
  from importlib.metadata import version
  report['simulator_version']=version('isaacsim')
  if not omni.usd.get_context().open_stage(str(root/'scene.usda')):raise RuntimeError('USD stage did not open')
  for _ in range(40):app.update()
  stage=omni.usd.get_context().get_stage()
  assert UsdGeom.GetStageMetersPerUnit(stage)==1. and UsdGeom.GetStageUpAxis(stage)=='Z'
  collider_paths=[str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
  assert len(collider_paths)==5
  world=World(stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=1/30,backend='numpy',device='cpu',physics_prim_path='/World/PhysicsScene')
  plans=json.loads((root/'asset_validation.json').read_text())['drop_probe_plan']
  if not plans:raise RuntimeError('No validated drop probes')
  spheres=[]
  for i,plan in enumerate(plans):
   position=np.array([*plan['xy'],plan['surface_z']+plan['radius_m']+plan['drop_height_m']])
   cube=plan['target']!='FloorSupport' and args.furniture_probe=='cube'
   plan['probe_shape']='cube' if cube else 'sphere'
   kwargs=dict(prim_path=f'/World/Validation/Probe_{i}',name=f'probe_{i}',position=position,mass=.1,color=np.array([.9,.15,.05]))
   obj=world.scene.add(DynamicCuboid(size=2*plan['radius_m'],**kwargs) if cube else DynamicSphere(radius=plan['radius_m'],**kwargs))
   PhysxSchema.PhysxContactReportAPI.Apply(obj.prim).CreateThresholdAttr().Set(0.)
   PhysxSchema.PhysxRigidBodyAPI.Apply(obj.prim).CreateEnableCCDAttr().Set(True)
   spheres.append(obj)
  contacts={f'/World/Validation/Probe_{i}':set() for i in range(len(spheres))}
  def callback(headers,data):
   for header in headers:
    a=str(PhysicsSchemaTools.intToSdfPath(header.collider0));b=str(PhysicsSchemaTools.intToSdfPath(header.collider1))
    if a in contacts:contacts[a].add(b)
    if b in contacts:contacts[b].add(a)
  subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(callback)
  world.reset()
  query=omni.physx.get_physx_scene_query_interface()
  ray_checks=[]
  for plan in plans:
   result=query.raycast_closest(tuple([*plan['xy'],plan['surface_z']+.15]),(0.,0.,-1.),.5)
   ray_checks.append({'target':plan['target'],'hit':bool(result.get('hit')),'collider':str(result.get('collision','')),'position':list(result.get('position',[]))})
  samples=[]
  for step in range(round(args.seconds*120)):
   world.step(render=False)
   if step%10==0:samples.append([obj.get_world_pose()[0].tolist() for obj in spheres])
  probes=[]
  for i,(obj,plan) in enumerate(zip(spheres,plans)):
   position=obj.get_world_pose()[0];velocity=obj.get_linear_velocity();path=f'/World/Validation/Probe_{i}'
   contact_list=sorted(contacts[path]);target=f'/World/Room/{plan["target"]}'
   target_contact=any(x==target or x.startswith(target+'/') for x in contact_list)
   finite=bool(np.isfinite(position).all() and np.isfinite(velocity).all())
   support_ok=bool(position[2]>=plan['radius_m']-.015)
   settled=bool(np.linalg.norm(velocity)<.05)
   floor_height_ok=abs(float(position[2])-plan['radius_m'])<.012 if plan['target']=='FloorSupport' else True
   surface_support_ok=abs(float(position[2])-plan['surface_z']-plan['radius_m'])<.025 if plan['probe_shape']=='cube' else True
   probes.append({**plan,'final_position':position.tolist(),'final_velocity':velocity.tolist(),'contact_paths':contact_list,
                  'target_contact_recorded':target_contact,'above_floor':support_ok,'settled':settled,'floor_height_ok':bool(floor_height_ok),'resting_on_target_height':bool(surface_support_ok),
                  'passed':bool(finite and target_contact and support_ok and settled and floor_height_ok and surface_support_ok)})
  report.update(status='passed' if all(p['passed'] for p in probes) and all(r['hit'] for r in ray_checks) else 'failed',
                physics_steps=round(args.seconds*120),colliders=collider_paths,raycasts=ray_checks,probes=probes,
                no_test_objects_saved_to_scene=True,scope='Static room loading, collider cooking, ray queries, and sampled ground sphere and furniture cube contacts. Spherical probes can roll on curved surfaces; their earlier diagnostic run is retained separately. Not a robot policy or sim-to-real validation.')
  (root/'isaac_probe_trajectories.json').write_text(json.dumps({'sample_dt':10/120,'positions':samples},indent=2)+'\n')
  world.stop();subscription=None
 except Exception as exc:
  report.update(status='failed',error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
 finally:
  (root/'isaac_validation.json').write_text(json.dumps(report,indent=2)+'\n')
  manifest=json.loads((root/'manifest.json').read_text());manifest['isaac_runtime_validation']=report['status'];manifest['isaac_validation_report']='isaac_validation.json';(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  print('ISAAC_VALIDATION_RESULT',json.dumps(report),flush=True)
  app.close()
 if report['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
