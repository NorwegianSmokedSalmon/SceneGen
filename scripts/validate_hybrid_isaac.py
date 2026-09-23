#!/usr/bin/env python3
"""Validate actual Isaac rendering of Gaussian background plus generated meshes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=ROOT/'results/scene_gen_room/hybrid/scene.usda')
    args = parser.parse_args()
    scene = args.scene.resolve()
    output = scene.parent
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 779, 'height': 519, 'multi_gpu': False,
                         'renderer': 'RaytracedLighting', 'enable_crashreporter': False,
                         'extra_args': ['--/app/settings/persistent=false', '--/app/useFabricSceneDelegate=true']})
    report = {'scene': str(scene), 'passed': False}
    try:
        import carb
        import numpy as np
        from PIL import Image, ImageDraw
        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom, UsdPhysics
        context = omni.usd.get_context()
        assert context.open_stage(str(scene))
        for _ in range(80): app.update()
        stage = context.get_stage()
        for key, value in stage.GetRootLayer().customLayerData.get('renderSettings', {}).items():
            carb.settings.get_settings().set('/'+key.replace(':', '/'), value)
        meshes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        volumes = [str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == 'Volume']
        colliders = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert len(meshes) == 2 and all(p.startswith('/World/Objects/') for p in meshes)
        assert len(volumes) == 1 and volumes[0].startswith('/World/BackgroundGS/')
        assert len(colliders) == 3 and not any(p.startswith('/World/BackgroundGS/') for p in colliders)
        bg = UsdGeom.Imageable(stage.GetPrimAtPath('/World/BackgroundGS'))
        obj = UsdGeom.Imageable(stage.GetPrimAtPath('/World/Objects'))
        rows = []
        records = []
        for index in [0, 120]:
            product = rep.create.render_product(f'/World/Cameras/RecordedView_{index}', (779, 519))
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach([product])
            images = {}
            for mode, show_background, show_objects in [('hybrid', True, True), ('objects_only', False, True), ('background_only', True, False)]:
                bg.GetVisibilityAttr().Set('inherited' if show_background else 'invisible')
                obj.GetVisibilityAttr().Set('inherited' if show_objects else 'invisible')
                for _ in range(80): app.update()
                rep.orchestrator.step(rt_subframes=8, pause_timeline=True)
                data = np.asarray(annotator.get_data())
                if data.shape != (519, 779, 4):
                    raise RuntimeError(f'Invalid render buffer {data.shape}')
                rgb = data[:, :, :3].copy()
                images[mode] = rgb
                Image.fromarray(rgb).save(output/f'isaac_{mode}_{index}.png')
                print(f'Captured camera {index}: {mode}', flush=True)
            bg_change = np.abs(images['hybrid'].astype(float)-images['objects_only'].astype(float)).mean(axis=2)
            obj_change = np.abs(images['hybrid'].astype(float)-images['background_only'].astype(float)).mean(axis=2)
            record = dict(camera=index, background_changed_fraction=float((bg_change>5).mean()),
                          objects_changed_pixels=int((obj_change>5).sum()), background_rgb_std=float(images['background_only'].std()))
            record['passed'] = record['background_changed_fraction']>.2 and record['objects_changed_pixels']>500 and record['background_rgb_std']>10
            records.append(record)
            row = Image.new('RGB', (779*3, 549), 'white')
            for j, mode in enumerate(['hybrid', 'background_only', 'objects_only']):
                row.paste(Image.fromarray(images[mode]), (779*j, 30))
                ImageDraw.Draw(row).text((779*j+10, 8), f'Camera {index} | {mode}', fill='black')
            rows.append(row)
            annotator.detach([product.path])
            product.destroy()
        bg.GetVisibilityAttr().Set('inherited')
        obj.GetVisibilityAttr().Set('inherited')
        board = Image.new('RGB', (779*3, 549*len(rows)), 'white')
        for i, row in enumerate(rows): board.paste(row, (0, 549*i))
        board.save(output/'isaac_hybrid_comparison.jpg', quality=92)
        report.update(meshes=meshes, gaussian_volumes=volumes, colliders=colliders, views=records,
                      passed=all(r['passed'] for r in records),
                      scope='Actual Isaac Sim 5.1 rendered RGB visibility ablation; static scene structure. Not a dynamic-physics certification.')
        if not report['passed']:
            raise RuntimeError('The rendered Gaussian background or object mesh was not visible')
    except BaseException as exc:
        import traceback
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        (output/'isaac_hybrid_validation.json').write_text(json.dumps(report, indent=2)+'\n')
        app.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
