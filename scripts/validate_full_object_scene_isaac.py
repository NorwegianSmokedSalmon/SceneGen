#!/usr/bin/env python3
"""Open, render and step the complete static object scene in the native Isaac runtime."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,default=ROOT/'results/scene_gen_room/full_objects/scene/scene.usda');a=p.parse_args();scene=a.scene.resolve();out=scene.parent;sys.argv=[sys.argv[0]]
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':960,'height':640,'multi_gpu':False,'renderer':'RaytracedLighting','enable_crashreporter':False,'extra_args':['--/app/settings/persistent=false','--/app/useFabricSceneDelegate=true']})
    report={'scene':str(scene),'passed':False}
    try:
        import carb,numpy as np
        from PIL import Image,ImageDraw
        import omni.usd,omni.timeline
        import omni.replicator.core as rep
        from pxr import UsdGeom,UsdPhysics
        context=omni.usd.get_context();assert context.open_stage(str(scene))
        for _ in range(100):app.update()
        stage=context.get_stage();meta=stage.GetRootLayer().customLayerData
        for key,value in meta.get('renderSettings',{}).items():carb.settings.get_settings().set('/'+key.replace(':','/'),value)
        prims=list(stage.Traverse());meshes=[str(p.GetPath()) for p in prims if p.IsA(UsdGeom.Mesh)];colliders=[str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert len(meshes)==int(meta['expectedMeshCount']);assert len(colliders)==int(meta['expectedColliderCount'])
        assert all(p.startswith('/World/Objects/') for p in meshes)
        assert not any(p.IsA(UsdGeom.Points) or p.GetTypeName()=='Volume' for p in prims)
        assert not stage.GetPrimAtPath('/World/BackgroundGS')
        views=[];rows=[]
        for name in ['Overview','RecordedView_0','RecordedView_120','RecordedView_270']:
            product=rep.create.render_product('/World/Cameras/'+name,(960,640));ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product])
            for _ in range(50):app.update()
            rep.orchestrator.step(rt_subframes=8,pause_timeline=True);data=np.asarray(ann.get_data());assert data.shape==(640,960,4),data.shape
            rgb=data[...,:3].copy();assert rgb.std()>5,'Blank scene render';Image.fromarray(rgb).save(out/f'isaac_{name}.png')
            row=Image.new('RGB',(960,668),'white');row.paste(Image.fromarray(rgb),(0,28));ImageDraw.Draw(row).text((10,8),name+' | independent generated object meshes',fill='black');rows.append(row)
            views.append({'camera':name,'rgb_std':float(rgb.std()),'image':f'isaac_{name}.png'});print('RENDERED',name,flush=True);ann.detach([product.path]);product.destroy()
        timeline=omni.timeline.get_timeline_interface();timeline.play()
        for _ in range(30):app.update()
        stepped=float(timeline.get_current_time());timeline.stop();assert stepped>0
        board=Image.new('RGB',(1920,1336),'white')
        for j,row in enumerate(rows):board.paste(row,((j%2)*960,(j//2)*668))
        board.save(out/'scene_preview.jpg',quality=90)
        report.update(passed=True,mesh_count=len(meshes),collider_count=len(colliders),point_cloud_count=0,gaussian_volume_count=0,views=views,physics_timeline_seconds=stepped,
                      scope='Isaac Sim native stage load, rendered visibility, static collision scene startup and timeline stepping; no dynamic object stability or contact accuracy certification')
    except BaseException as exc:
        import traceback
        report.update(error=str(exc),traceback=traceback.format_exc());raise
    finally:
        (out/'isaac_validation.json').write_text(json.dumps(report,indent=2)+'\n');app.close()
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
