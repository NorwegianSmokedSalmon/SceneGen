#!/usr/bin/env python3
"""Compose original background Gaussians and independently generated object meshes.

No room meshing, TSDF, or room collision reconstruction is performed. Run in the
project WorldSculpt environment with NVIDIA 3DGRUT v1.1.0 checked out locally.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / '.cache/scene_gen/vendor/3dgrut'
SPECS = [(0, 'Armchair', 'reconstruction_train'), (1, 'SideTable', 'reconstruction')]


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def export_background(splats, keep, assets):
    sys.path.insert(0, str(VENDOR))
    from threedgrut.export.accessor import GaussianAttributes, ModelCapabilities
    from threedgrut.export.adapter import AttributesExportAdapter
    from threedgrut.export.usd.nurec.exporter import NuRecExporter
    p = {k: v[keep].contiguous() for k, v in splats.items()}
    torch.save({'splats': p, 'step': 6999}, assets / 'background_gs.pt')
    attrs = GaussianAttributes(
        positions=p['means'].numpy(), rotations=p['quats'].numpy(),
        scales=p['scales'].numpy(), densities=p['opacities'][:, None].numpy(),
        albedo=p['sh0'][:, 0].numpy(), specular=p['shN'].flatten(1).numpy())
    caps = ModelCapabilities(has_spherical_harmonics=True, sh_degree=3,
                             num_gaussians=len(p['means']), is_surfel=False)
    NuRecExporter().export(AttributesExportAdapter(attrs, caps, device='cpu'),
                          assets / 'background_gs.usdz', apply_coordinate_transform=False)
    # NuRec stores float16 Gaussian data; the .pt retains exact source precision.
    return len(p['means'])


def stage_units(stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)


def vertex_material(stage):
    material = UsdShade.Material.Define(stage, '/Objects/Materials/ObservedColor')
    shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + '/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.85)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
    reader = UsdShade.Shader.Define(stage, str(material.GetPath()) + '/Color')
    reader.CreateIdAttr('UsdPrimvarReader_float3')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('displayColor')
    reader.CreateOutput('result', Sdf.ValueTypeNames.Float3)
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), 'result')
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return material


def export_objects(args, splats, labels, transform, assets):
    stage = Usd.Stage.CreateNew(str(assets / 'objects.usdc'))
    stage_units(stage)
    root = UsdGeom.Xform.Define(stage, '/Objects')
    stage.SetDefaultPrim(root.GetPrim())
    material = vertex_material(stage)
    records = []
    glb_scene = trimesh.Scene()
    z_to_y = np.array([[1., 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    for ident, name, variant in SPECS:
        directory = args.comparison / 'direct' / f'instance_{ident}'
        source = directory / variant / 'mesh_world.glb'
        run = json.loads((source.parent / 'run.json').read_text())
        if run['status'] != 'succeeded':
            raise ValueError(f'Object generation is incomplete: {source}')
        meta = json.loads((directory / 'transforms.json').read_text())
        mesh = trimesh.load(source, force='mesh', process=False)
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError(f'Invalid generated mesh: {source}')
        original = mesh.vertices.copy()
        selected = labels == ident
        tree = cKDTree(splats['means'][selected].numpy())
        _, nearest = tree.query(original, workers=8)
        dc = splats['sh0'][selected, 0].numpy()
        colors = np.clip(.5 + .28209479177387814 * dc[nearest], 0, 1)
        mesh.visual.vertex_colors = np.column_stack([colors * 255, np.full(len(colors), 255)]).astype('uint8')
        mesh.apply_transform(transform)
        world_vertices = mesh.vertices.copy()
        center = transform[:3, :3] @ np.asarray(meta['offset']) + transform[:3, 3]
        mesh.vertices -= center
        object_root = UsdGeom.Xform.Define(stage, f'/Objects/{name}')
        object_root.AddTranslateOp().Set(Gf.Vec3d(*center))
        object_root.GetPrim().SetCustomDataByKey('instanceId', ident)
        object_root.GetPrim().SetCustomDataByKey('geometrySource', 'WorldSculpt calibrated multiview generation')
        visual = UsdGeom.Mesh.Define(stage, f'/Objects/{name}/Visual')
        visual.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, np.float32)))
        visual.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, np.int32)))
        visual.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces, np.int32).ravel()))
        visual.CreateSubdivisionSchemeAttr('none')
        visual.CreateDoubleSidedAttr(True)
        visual.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.bounds, np.float32)))
        visual.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_normals, np.float32)))
        visual.SetNormalsInterpolation('vertex')
        linear = np.where(colors <= .04045, colors / 12.92, ((colors + .055) / 1.055) ** 2.4)
        visual.CreateDisplayColorPrimvar('vertex').Set(Vt.Vec3fArray.FromNumpy(linear.astype(np.float32)))
        UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
        # Static triangle collisions on the object itself; no background collider.
        UsdPhysics.CollisionAPI.Apply(visual.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(visual.GetPrim()).CreateApproximationAttr('none')
        mesh.export(assets / f'{name.lower()}_local_zup.ply')
        placed = mesh.copy()
        placed.vertices = world_vertices
        placed.apply_transform(z_to_y)
        placed.export(assets / f'{name.lower()}_placed.glb')
        glb_scene.add_geometry(placed, node_name=name, geom_name=name)
        inverse = np.linalg.inv(transform)
        roundtrip = (mesh.vertices + center) @ inverse[:3, :3].T + inverse[:3, 3]
        error = float(np.abs(roundtrip - original).max())
        if error > 1e-6:
            raise ValueError('Object placement does not round-trip to source coordinates')
        records.append(dict(instance_id=ident, name=name, source=str(source.relative_to(ROOT)),
                            source_sha256=digest(source), model=run['model'], sampler=run['sampler'],
                            faces=len(mesh.faces), vertices=len(mesh.vertices), position_m=center.tolist(),
                            bounds_m=[world_vertices.min(0).tolist(), world_vertices.max(0).tolist()],
                            placement_roundtrip_max_error=error, source_gaussians_removed=int(selected.sum()),
                            color_source='Nearest labeled Gaussian DC color; no new texture generation',
                            is_watertight=bool(mesh.is_watertight)))
        print(f'{name}: placed {len(mesh.faces)} generated triangles', flush=True)
    stage.GetRootLayer().Save()
    glb_scene.export(args.output / 'objects_placed.glb')
    return records


def compose_stage(args, coordinates, records, include_background=True):
    filename = 'scene.usda' if include_background else 'objects_only.usda'
    stage = Usd.Stage.CreateNew(str(args.output / filename))
    stage_units(stage)
    root = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(root.GetPrim())
    transform = np.asarray(coordinates['world_to_simulation'])
    if include_background:
        bg = UsdGeom.Xform.Define(stage, '/World/BackgroundGS')
        bg.GetPrim().GetReferences().AddReference('assets/background_gs.usdz')
        bg.AddTransformOp().Set(Gf.Matrix4d(transform.T.tolist()))
    obj = UsdGeom.Xform.Define(stage, '/World/Objects')
    obj.GetPrim().GetReferences().AddReference('assets/objects.usdc')
    physics = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    # One invisible analytic box supports objects. It is not a reconstructed mesh.
    bounds = np.asarray(coordinates['bounds_simulation_m'])
    floor = UsdGeom.Cube.Define(stage, '/World/FloorSupport')
    floor.CreateSizeAttr(1.)
    floor.AddTranslateOp().Set(Gf.Vec3d(*bounds.mean(0)[:2], -.025))
    floor.AddScaleOp().Set(Gf.Vec3f(*(bounds[1] - bounds[0])[:2], .05))
    floor.CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim()).CreateCollisionEnabledAttr(True)
    light = UsdLux.DomeLight.Define(stage, '/World/Lighting')
    light.CreateIntensityAttr(400.)
    views = torch.load(args.tracking, map_location='cpu', weights_only=False)['view_data']
    for index in [0, 120, 200, 270]:
        v = views[index]
        pose = v['camtoworld'].numpy().copy()
        pose[:3, 3] = transform[:3, :3] @ pose[:3, 3] + transform[:3, 3]
        pose[:3, :3] = transform[:3, :3] / coordinates['meters_per_scene_unit'] @ pose[:3, :3]
        pose = pose @ np.diag([1., -1., -1., 1.])
        camera = UsdGeom.Camera.Define(stage, f'/World/Cameras/RecordedView_{index}')
        camera.AddTransformOp().Set(Gf.Matrix4d(pose.T.tolist()))
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(36. * v['height'] / v['width'])
        camera.CreateFocalLengthAttr(float(v['K'][0, 0]) * 36. / v['width'])
        camera.CreateClippingRangeAttr(Gf.Vec2f(.02, 100.))
    stage.GetRootLayer().customLayerData = {
        'sceneMode': ('gaussian_background_generated_object_meshes' if include_background
                      else 'generated_object_meshes_only'),
        'expectedGaussianVolumeCount': int(include_background),
        'expectedMeshCount': len(records), 'expectedColliderCount': len(records) + 1,
        'selectionRoot': '/World/Objects',
        'renderSettings': {'rtx:rendermode': 'RaytracedLighting',
                           'rtx:post:histogram:enabled': False,
                           'rtx:post:registeredCompositing:invertToneMap': True,
                           'rtx:post:registeredCompositing:invertColorCorrection': True,
                           'rtx:post:tonemap:op': 2}}
    stage.GetRootLayer().Save()
    check = Usd.Stage.Open(str(args.output / filename))
    meshes = [str(p.GetPath()) for p in check.Traverse() if p.IsA(UsdGeom.Mesh)]
    volumes = [str(p.GetPath()) for p in check.Traverse() if p.GetTypeName() == 'Volume']
    colliders = [str(p.GetPath()) for p in check.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(meshes) == len(records) and all(p.startswith('/World/Objects/') for p in meshes)
    assert len(volumes) == int(include_background) and len(colliders) == len(records) + 1
    if not include_background:
        assert not check.GetPrimAtPath('/World/BackgroundGS')
        assert not any(p.IsA(UsdGeom.Points) for p in check.Traverse())
        assert not any('background_gs' in str(layer.identifier) for layer in check.GetUsedLayers())
    return dict(meshes=meshes, gaussian_volumes=volumes, colliders=colliders, background_mesh_count=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'results/scene_gen_room/ckpts/ckpt_6999_rank0.pt')
    p.add_argument('--labels', type=Path, default=ROOT / 'data/mipnerf360_indoor/room/cluster_result/instance_labels_dense.npy')
    p.add_argument('--tracking', type=Path, default=ROOT / 'data/mipnerf360_indoor/room/cluster_result/gauscluster_tracking_data.pt')
    p.add_argument('--comparison', type=Path, default=ROOT / 'results/scene_gen_room/multiview_comparison')
    p.add_argument('--coordinates', type=Path, default=ROOT / 'results/scene_gen_room/simulation/coordinates.json')
    p.add_argument('--output', type=Path, default=ROOT / 'results/scene_gen_room/hybrid')
    args = p.parse_args()
    torch.set_num_threads(8)
    assets = args.output / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    coordinates = json.loads(args.coordinates.read_text())
    splats = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['splats']
    labels = np.load(args.labels)
    assert len(labels) == len(splats['means'])
    keep = ~np.isin(labels, [r[0] for r in SPECS])
    np.save(assets / 'background_source_indices.npy', np.flatnonzero(keep))
    count = export_background(splats, keep, assets)
    records = export_objects(args, splats, labels, np.asarray(coordinates['world_to_simulation']), assets)
    structure = compose_stage(args, coordinates, records)
    objects_only_structure = compose_stage(args, coordinates, records, include_background=False)
    save_json(args.output / 'coordinates.json', coordinates)
    manifest = dict(scene_mode='gaussian_background_generated_object_meshes',
                    source_checkpoint=str(args.checkpoint.relative_to(ROOT)),
                    source_checkpoint_sha256=digest(args.checkpoint),
                    labels_sha256=digest(args.labels), source_gaussians=len(labels),
                    background_gaussians=count, removed_gaussians=int((~keep).sum()),
                    replaced_instance_ids=[r[0] for r in SPECS], retained_instance_ids=sorted(set(labels[keep].tolist())),
                    objects=records, structure=structure, background_meshing=False,
                    default_isaac_scene='objects_only.usda', objects_only_structure=objects_only_structure,
                    background_precision='original float32 in PT; official NuRec conversion uses float16',
                    converter_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=VENDOR, text=True).strip(),
                    scale_is_estimated=coordinates['scale_is_estimated'],
                    physics='Static object triangle colliders and one invisible analytic floor box. Background GS has no collisions.',
                    limitations=['Generated meshes retain known reconstruction defects; this is a composition correction.',
                                 'Footstool and unselected scene elements remain original Gaussians.',
                                 'No mesh or collision geometry was generated for the room.',
                                 'Removal follows dense labels; segmentation errors can leave boundary remnants.'])
    save_json(args.output / 'manifest.json', manifest)
    print(json.dumps(structure, indent=2), flush=True)


if __name__ == '__main__':
    main()
