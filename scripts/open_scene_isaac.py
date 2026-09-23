#!/usr/bin/env python3
"""Open independently generated object meshes in Isaac Sim; omit the 3DGS background.

Run using Isaac Sim's Python. Any NVIDIA license prompt is left to the user.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]


class ConsoleLog:
    def __init__(self,stream,log):self.stream=stream;self.log=log
    def write(self,message):
        self.log.write(message);self.log.flush()
        return self.stream.write(message)
    def flush(self):self.stream.flush();self.log.flush()
    def __getattr__(self,name):return getattr(self.stream,name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,default=ROOT/'results/scene_gen_room/full_objects/scene/scene.usda')
    parser.add_argument('--camera',default=None)
    args=parser.parse_args();scene=args.scene.resolve()
    if not scene.is_file():raise FileNotFoundError(scene)
    cache=ROOT/'.cache/scene_gen';cache.mkdir(parents=True,exist_ok=True)
    status_path=cache/'isaac_gui_status.json'
    log=(cache/'isaac_gui.log').open('w',buffering=1)
    sys.stdout=ConsoleLog(sys.stdout,log);sys.stderr=ConsoleLog(sys.stderr,log)
    status={'pid':os.getpid(),'scene':str(scene),'camera_requested':args.camera,'started_utc':datetime.now(timezone.utc).isoformat()}
    def update(state,**extra):
        status.update(state=state,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
        temporary=status_path.with_suffix('.tmp');temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(status_path)
    update('initializing_runtime')
    app=None
    sys.argv=[sys.argv[0]]
    try:
        # This import may present NVIDIA's own EULA prompt on the attached terminal.
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':False,'hide_ui':False,'width':1280,'height':800,
                           'window_width':1440,'window_height':1000,'multi_gpu':False,
                           'renderer':'RaytracedLighting','enable_crashreporter':False,
                           'extra_args':['--/app/settings/persistent=false', '--/app/useFabricSceneDelegate=true']})
        import omni.usd
        import omni.kit.app
        from pxr import UsdGeom,UsdPhysics
        from omni.kit.viewport.utility import get_active_viewport
        update('opening_scene')
        context=omni.usd.get_context()
        kit=omni.kit.app.get_app()
        if not context.open_stage(str(scene)):raise RuntimeError('Isaac Sim could not open scene.usda')
        for _ in range(120):
            if not kit.is_running() or app.is_exiting():raise RuntimeError('Window closed before scene load completed')
            app.update()
        stage=context.get_stage()
        metadata=stage.GetRootLayer().customLayerData
        hybrid=metadata.get('sceneMode')=='gaussian_background_generated_object_meshes'
        objects_only=metadata.get('sceneMode')=='generated_object_meshes_only'
        selection_root=metadata.get('selectionRoot','/World/Room')
        if not stage.GetPrimAtPath(selection_root):raise RuntimeError('Scene objects are missing from the loaded stage')
        meshes=[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        colliders=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        expected_meshes=int(metadata.get('expectedMeshCount',8))
        expected_colliders=int(metadata.get('expectedColliderCount',5))
        if len(meshes)!=expected_meshes or len(colliders)!=expected_colliders:
            raise RuntimeError(f'Incomplete scene: {len(meshes)} meshes, {len(colliders)} colliders')
        volumes=[str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName()=='Volume']
        if hybrid and (len(volumes)!=1 or any(not p.startswith('/World/Objects/') for p in meshes)):
            raise RuntimeError('Hybrid scene must contain a Gaussian background and only object meshes')
        if objects_only and (volumes or stage.GetPrimAtPath('/World/BackgroundGS')
                             or any(p.IsA(UsdGeom.Points) for p in stage.Traverse())
                             or any(not p.startswith('/World/Objects/') for p in meshes)):
            raise RuntimeError('Objects-only scene must contain only generated object meshes, with no Gaussian background')
        import carb
        for key,value in metadata.get('renderSettings',{}).items():
            carb.settings.get_settings().set('/'+key.replace(':','/'),value)
        viewport=get_active_viewport()
        if viewport is None:raise RuntimeError('Isaac Sim has no active desktop viewport')
        args.camera=args.camera or metadata.get('defaultCamera','/World/Cameras/RecordedView_0')
        if not stage.GetPrimAtPath(args.camera):raise RuntimeError('Requested viewing camera is missing')
        viewport.camera_path=args.camera
        # Keep the observed camera projection while fitting its image in the viewport.
        context.get_selection().set_selected_prim_paths([],False)
        for _ in range(30):app.update()
        update('ready',loaded_scene=str(stage.GetRootLayer().realPath),mesh_count=len(meshes),collider_count=len(colliders),
               active_camera=str(viewport.camera_path),meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),up_axis=str(UsdGeom.GetStageUpAxis(stage)),
               scene_mode=metadata.get('sceneMode','legacy_full_room_mesh'),gaussian_volumes=volumes)
        print('ISAAC_ROOM_OPENED',scene,flush=True)
        print('Scene loaded: generated object meshes only; no 3DGS background.' if objects_only else
              'Scene loaded: Gaussian background and generated object meshes.' if hybrid else
              'Legacy full-mesh room is loaded.',flush=True)
        while kit.is_running() and not app.is_exiting():app.update()
        update('closed')
    except BaseException as exc:
        update('error',error=str(exc),error_type=type(exc).__name__)
        traceback.print_exc()
        raise
    finally:
        if app is not None:app.close()


if __name__=='__main__':main()
